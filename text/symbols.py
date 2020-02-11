""" from https://github.com/keithito/tacotron """

'''
Defines the set of symbols used in text input to the model.

The default is a set of ASCII characters that works well for English or text that has been run through Unidecode. For other data, you can modify _characters. See TRAINING_DATA.md for details. '''
from text import cmudict
#_eos = '*' #Este lo he utilizado para la voz de niño, hay que comentarlo para el modelo de aintzane y kiko
_pad        = '_'
_punctuation = '!\'(),.:;?¡¿ ' #Castellano y Euskera
#_punctuation = '!\'(),.:;? ' #Ingles
_special = '-'
#_letters = 'AÁBCDEÉFGHIÍJKLMNOÓPQRSTUÚVWXYZaábcdeéfghiíjklmnoópqrstuúvwxyzñÑ' #Castellano
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzñÑ' #Euskera
#_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' #Inglés

# Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
_arpabet = ['@' + s for s in cmudict.valid_symbols]

# Export all symbols:
symbols = [_pad] + list(_special) + list(_punctuation) + list(_letters) + _arpabet #+ [_eos]
