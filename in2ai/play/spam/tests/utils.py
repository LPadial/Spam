import re
match = re.search(r'^((?:sp|h)am) .*?inmail\.(\d{1,5})$', 'spam ../data/inmail.123456')
print(match)
#Spam inmail.
#((?:Sp|h)am) .*?inmail\\(\\d{1,5}).