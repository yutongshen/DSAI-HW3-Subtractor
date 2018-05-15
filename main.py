import numpy as np
import random
import pickle
import os.path


# Parameters Config
SUB      = 0
SUB_ADD  = 1
MULTIPLY = 2

DIGITS          = 4
TRAINING_SIZE   = 18000
VALIDATION_SIZE = 2000
TESTING_SIZE    = 60000
ITERATION       = 100
TOTAL_SIZE      = TRAINING_SIZE + VALIDATION_SIZE + TESTING_SIZE
GEN_TYPE        = None
MAXLEN          = None
ANS_DIGITS      = None
chars           = None
ct              = None

def set_gen_type(gen_type):
    global GEN_TYPE
    global ANS_DIGITS
    global chars
    global ct

    GEN_TYPE = gen_type

    ANS_DIGITS = {
        SUB: DIGITS + 1,
        SUB_ADD: DIGITS + 1,
        MULTIPLY: 2 * DIGITS
    }.get(GEN_TYPE, DIGITS + 1)

    chars = {
        SUB: '0123456789- ',
        SUB_ADD: '0123456789+- ',
        MULTIPLY: '0123456789* '
    }.get(GEN_TYPE, '0123456789+-* ')

    ct = CharacterTable(chars)

class CharacterTable:
    def __init__(self, chars):
        self.chars  = list(chars)
        self.len    = len(chars)
        self.encode = {}
        for i, key in enumerate(self.chars):
            self.encode[key] = np.zeros(self.len, np.float32)
            self.encode[key][i] = 1.
        
    def encoder(self, C):
        result = np.zeros((len(C), self.len))
        for i, c in enumerate(C):
            try:
                result[i] = self.encode[c]
            except:
                pass
        return result
        
    def decoder(self, x):
        x = x.argmax(axis=-1)
        return ''.join(self.chars[i] for i in x)

# Data Generation

def generation(arg):
    questions = []
    expected = []
    seen = set()
    operator = {
        SUB: ['-'],
        SUB_ADD: ['-', '+'],
        MULTIPLY: ['*']
    }
    ans_switcher = {
        '+': lambda a, b: a + b,
        '-': lambda a, b: a - b,
        '*': lambda a, b: a * b
    }
    ops = operator.get(GEN_TYPE, [None])
    print('Generating data...')
    while len(questions) < TOTAL_SIZE:
        f = lambda: random.choice(range(10 ** random.choice(range(1, DIGITS + 1))))
        g = lambda: random.choice(ops)
        a, b, op = f(), f(), g()
        if op == '-':
            a, b = sorted((a, b), reverse=True)
        key = tuple((a, b, op))
        if key in seen:
            continue
        seen.add(key)
        # query = '{}{}{}'.format(a, op, b).ljust(MAXLEN)
        query = str(a).rjust(DIGITS) + op + str(b).rjust(DIGITS)
        ans_funct = ans_switcher.get(op, lambda a, b: float('NAN'))
        ans = str(ans_funct(a, b)).rjust(ANS_DIGITS)
        questions.append(query)
        expected.append(ans)
    print('Total addition questions:', len(questions))
    
    print(questions[:10])
    print(expected[:10])

    # Processing
    
    x = np.zeros((len(questions), MAXLEN, len(chars)), np.float32)
    y = np.zeros((len(expected), ANS_DIGITS, len(chars)), np.float32)
    for i, sentence in enumerate(questions):
        x[i] = ct.encoder(sentence)
    for i, sentence in enumerate(expected):
        y[i] = ct.encoder(sentence)
    
    data = {}

    data['train_x'] = x[:TRAINING_SIZE]
    data['train_y'] = y[:TRAINING_SIZE]

    data['validation_x'] = x[TRAINING_SIZE:TRAINING_SIZE + VALIDATION_SIZE]
    data['validation_y'] = y[TRAINING_SIZE:TRAINING_SIZE + VALIDATION_SIZE]

    data['test_x'] = x[TRAINING_SIZE + VALIDATION_SIZE:]
    data['test_y'] = y[TRAINING_SIZE + VALIDATION_SIZE:]

    data['type'] = GEN_TYPE
    
    with open(arg.d, 'wb') as f:
        pickle.dump(data, f, -1)
        print('Save file successfully.')

# Model
# - Using sequence to sequence model
# - Encoder: bi-directional LSTM
# - Decoder: LSTM

def train(arg):
    if not os.path.exists(arg.d):
        print('Training data not exist.')
        return
    else:
        with open(arg.d, 'rb') as f:
            data = pickle.load(f)

        train_x = data['train_x']
        train_y = data['train_y']
                    
        validation_x = data['validation_x']
        validation_y = data['validation_y']
                    
        test_x = data['test_x']
        test_y = data['test_y']
        
        set_gen_type(data['type'])

    HIDDEN_SIZE = 256
    
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",category=FutureWarning)
        warnings.filterwarnings("ignore",category=UserWarning)
        import keras as K
        from keras.models import Sequential, Model
        from keras.layers.core import Dense, Activation, Lambda
        from keras.layers import Input, LSTM, TimeDistributed, RepeatVector, Reshape, Dropout, Bidirectional, Concatenate
        from keras.layers.normalization import BatchNormalization

    model = Sequential()
    
    encoder_inputs = Input(shape=(MAXLEN, len(chars)))
    encoder = Bidirectional(LSTM(HIDDEN_SIZE, return_state=True))
    encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder(encoder_inputs)
    
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    states = [state_h, state_c]
    
    # Set up the decoder, which will only process one timestep at a time.
    decoder_inputs = Reshape((1, HIDDEN_SIZE * 2))
    decoder_lstm = LSTM(HIDDEN_SIZE * 2, return_state=True)
    
    all_outputs = []
    inputs = decoder_inputs(encoder_outputs)
    
    first_decoder = True
    for _ in range(ANS_DIGITS):
        # Run the decoder on one timestep
        outputs, state_h, state_c = decoder_lstm(inputs,
        initial_state=states)
        
        # Reinject the outputs as inputs for the next loop iteration
        # as well as update the states
        states = [state_h, state_c]
        
        
        # Store the current prediction (we will concatenate all predictions later)
        outputs = Dense(len(chars), activation='softmax')(outputs)
        all_outputs.append(outputs)
    
    # Concatenate all predictions
    decoder_outputs = Concatenate()(all_outputs)
    decoder_outputs = Reshape((ANS_DIGITS, len(chars)))(decoder_outputs)
    decoder_outputs = Lambda(lambda x: x[:, ::-1])(decoder_outputs)
    
    # Define and compile model as previously
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    model.summary()
    
    batch_size = int(len(train_x) / 128 / 100) * 100

    if batch_size == 0:
        batch_size = 100
    
    model.fit(train_x, train_y, 
              batch_size=batch_size, epochs=ITERATION, 
              verbose=1, validation_data=[validation_x, validation_y])

    model.save(arg.m)
    print('save model successfully')

def check_error(model, x, y):
    err_list = []
    pred = model.predict(x)
    size = len(x)
    for i in range(size):
        y_str    = ct.decoder(y[i])
        pred_str = ct.decoder(pred[i])
        if y_str != pred_str:
            err_list.append(ct.decoder(x[i]) + ' = ' + pred_str + ' ' + y_str)
    return err_list

def report(arg, obj):
    if not os.path.exists(arg.d):
        print('Data not exist.')
        return
    else:
        with open(arg.d, 'rb') as f:
            data = pickle.load(f)

    if obj == 'acc':
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",category=FutureWarning)
            from keras.models import load_model

        if not os.path.exists(arg.m):
            print('Model file not exist.')
            return
        else:
            set_gen_type(data['type'])
            model = load_model(arg.m)
            eva = model.evaluate(data['train_x'], data['train_y'], verbose=False)
            print('Training Data:')
            print('Size:', len(data['train_x']))
            print('Loss:', eva[0], 'Accuracy:', eva[1])
            err_list = check_error(model, data['train_x'], data['train_y'])
            print('ERROR:', len(err_list), '/', len(data['train_x']))
            if len(err_list):
                print('predict'.rjust(MAXLEN + ANS_DIGITS + 3), 'ans'.rjust(ANS_DIGITS))
                print('\n'.join(err_list))
            print()
            eva = model.evaluate(data['validation_x'], data['validation_y'], verbose=False)
            print('Validation Data:')
            print('Size:', len(data['validation_x']))
            print('Loss:', eva[0], 'Accuracy:', eva[1])
            err_list = check_error(model, data['validation_x'], data['validation_y'])
            print('ERROR:', len(err_list), '/', len(data['validation_x']))
            if len(err_list):
                print('predict'.rjust(MAXLEN + ANS_DIGITS + 3), 'ans'.rjust(ANS_DIGITS))
                print('\n'.join(err_list))
            print()
            eva = model.evaluate(data['test_x'], data['test_y'], verbose=False)
            print('Testing Data:')
            print('Size:', len(data['test_x']))
            print('Loss:', eva[0], 'Accuracy:', eva[1])
            err_list = check_error(model, data['test_x'], data['test_y'])
            print('ERROR:', len(err_list), '/', len(data['test_x']))
            if len(err_list):
                print('predict'.rjust(MAXLEN + ANS_DIGITS + 3), 'ans'.rjust(ANS_DIGITS))
                print('\n'.join(err_list))
            print()
    else:
        report_x = data[obj + '_x']
        report_y = data[obj + '_y']

        set_gen_type(data['type'])

        for i in range(len(report_x)):
            print(ct.decoder(report_x[i]), '=', ct.decoder(report_y[i]))
            
def test(arg):
    from keras.models import load_model

    if not os.path.exists(arg.m):
        print('Model file not exist.')
        return
    else:
        model = load_model(arg.m)

    while True:
        q = input('Please input test data or "exit": ')
        
        if q.upper() == 'EXIT':
            break;
        
        q_padding = q.ljust(MAXLEN)[:MAXLEN]
        test_x = ct.encoder(q_padding)
        pred_y = model.predict(test_x.reshape(-1, MAXLEN, len(chars)))
        print(q, '=', ct.decoder(pred_y[0]))

def help(arg = None):
    print('usage: python main.py [-o OPTION] [-t TYPE] [-d DATA] [-m MODEL]')
    print()
    print('general options:')
    print('  -h, --help                 show this help message and exit')
    print('')
    print('operational options:')
    print('  -o gen                     data generation')
    print('  -o train                   training model')
    print('  -o report_training_data    show all training data')
    print('  -o report_validation_data  show all validation data')
    print('  -o report_testing_data     show all testing data')
    print('  -o report_accuracy         show accuracy')
    print('  -o test                    input formula by self')
    print('')
    print('calculational options: (default: -t sub)')
    print('  -t sub                     subtraction')
    print('  -t sub_add                 subtraction mix with addition')
    print('  -t multiply                multiplication')
    print('')
    print('advance options:')
    print('  -d <DATA>                  input the path of training (or generation) data')
    print('                             (default: src/data.pkl)')
    print('  -m <MODEL>                 input the path of model')
    print('                             (default: src/my_model.h5)')
    print('')
    
import argparse

class MyParser(argparse.ArgumentParser):
    def format_help(self):
        help()
        return 

if __name__ == '__main__':
    
    # parser = argparse.ArgumentParser()
    parser = MyParser()

    parser.add_argument('-o',
                        default='unknown',
                        help='input operation.')

    parser.add_argument('-t',
                        default='sub',
                        help='input calculation type.')

    parser.add_argument('-d',
                        default='src/data.pkl',
                        help='input data.')

    parser.add_argument('-m',
                        default='src/my_model.h5',
                        help='input model.')

    args = parser.parse_args()

    GEN_TYPE = {
        'sub': SUB,
        'sub_add': SUB_ADD,
        'multiply': MULTIPLY
    }.get(args.t, -1)
    
    if GEN_TYPE != -1:
        MAXLEN = DIGITS + 1 + DIGITS

        set_gen_type(GEN_TYPE)


        switcher = {
            'gen':                    lambda arg: generation(arg),
            'train':                  lambda arg: train(arg),
            'report_training_data':   lambda arg: report(arg, 'train'),
            'report_validation_data': lambda arg: report(arg, 'validation'),
            'report_testing_data':    lambda arg: report(arg, 'test'),
            'report_accuracy':        lambda arg: report(arg, 'acc'),
            'test':                   lambda arg: test(arg)
        }

        func = switcher.get(args.o, lambda arg: help(arg))

        func(args)
    else:
        help()

