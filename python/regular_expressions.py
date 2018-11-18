
# coding: utf-8

# # Python regular expressions tutorial
# ___

# In[2]:


import re


# ## Simple

# In[99]:


text_to_search = '''
abcdefghijklmnopqurtuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
1234567890
Ha HaHa
MetaCharacters (Need to be escaped):
. ^ $ * + ? { } [ ] \ | ( )
coreyms.com
321-555-4321
123.555.1234
123*555*1234
800-555-1234
900-555-1234
Mr. Schafer
Mr Smith
Ms Davis
Mrs. Robinson
Mr. T

cat
mat
pat
bat
'''


# In[4]:


sentence = 'Start a sentence and then bring it to an end'

pattern = re.compile(r'start', re.I)

matches = pattern.search(sentence)

print(matches)


# In[16]:


pattern = re.compile(r'\bHa')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[24]:


pattern = re.compile(r'end$')
matches = pattern.finditer(sentence)
for match in matches:
    print(match)


# In[34]:


pattern = re.compile(r'[89]00[.-]\d\d\d[.-]\d\d\d\d')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[41]:


pattern = re.compile(r'[^b]at')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[102]:


pattern = re.compile(r'\d{3}.\d{3}.\d{4}')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[56]:


pattern = re.compile(r'Mr\.?\s\w+')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[60]:


pattern = re.compile(r'M(r|s|rs)\.?\s\w+')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[61]:


pattern = re.compile(r'(Mr|Ms|Mrs)\.?\s\w+')
matches = pattern.finditer(text_to_search)
for match in matches:
    print(match)


# In[96]:


pattern = re.compile(r'(Mr|Ms|Mrs)\.?\s\w+')
matches = pattern.findall(text_to_search)
for match in matches:
    print(match)


# In[104]:


pattern = re.compile(r'\d{3}.\d{3}.\d{4}')
matches = pattern.findall(text_to_search)
for match in matches:
    print(match)


# ###### match the string beginning

# In[108]:


pattern = re.compile(r'Start')
matches = pattern.match(sentence)
print(matches)


# ###### search through entire string

# In[110]:


pattern = re.compile(r'sentence')
matches = pattern.search(sentence)
print(matches)


# ###### flags

# In[114]:


pattern = re.compile(r'start', re.IGNORECASE)
matches = pattern.search(sentence)
print(matches)


# In[115]:


pattern = re.compile(r'start', re.I)
matches = pattern.search(sentence)
print(matches)


# ## Data.txt

# In[35]:


with open('data.txt') as f:
    contents = f.read()
    matches = pattern.finditer(contents)
    for match in matches:
        print(match)


# ## E-mails

# In[69]:


import re

emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

pattern = re.compile(r'[a-zA-Z0-9.-]+@[a-zA-Z-]+\.(com|edu|net)')

matches = pattern.finditer(emails)

for match in matches:
    print(match)


# In[71]:


import re

emails = '''
CoreyMSchafer@gmail.com
corey.schafer@university.edu
corey-321-schafer@my-work.net
'''

pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')

matches = pattern.finditer(emails)

for match in matches:
    print(match)


# In[ ]:


'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'


# ## URLs

# In[81]:


import re

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

pattern = re.compile(r'https?://(www\.)?\w+\.\w+')

# subbed_urls = pattern.sub(r'\2\3', urls)

# print(subbed_urls)

matches = pattern.finditer(urls)

for match in matches:
    print(match)


# In[89]:


import re

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')

# sibbed_urls = pattern.sub(r'\2\3', urls)
# print(subbed_urls)

matches = pattern.finditer(urls)

for match in matches:
    print(match.group(3))


# In[90]:


import re

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')

sibbed_urls = pattern.sub(r'\2\3', urls)
print(subbed_urls)


# In[95]:


import re

urls = '''
https://www.google.com
http://coreyms.com
https://youtube.com
https://www.nasa.gov
'''

pattern = re.compile(r'https?://(www\.)?(\w+)(\.\w+)')

matches = pattern.findall(urls)

for match in matches:
    print(match)


# In[74]:


subbed_urls


# In[ ]:


https?://(www\.)?(\w+)(\.\w+)

