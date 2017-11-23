
# coding: utf-8

# # Basics
# 
# Press SHIFT+ENTER TO EXECUTE A CELL OF CODE

# In[ ]:


##Comments begin with a '#' symbol.
print(1+1)
print(1.2*3.4)
print(3/2)
print(3.0/2)
a = 'test'
print(a)
b = 'sample\nstring'
print(b)


# In[2]:


c = ['1', 2, 'a', 3, 'hello']
print(c)
print("5%4",5%4) #remainder function.
print("11%7",11%7)
print("5>4  {}".format(5 > 4))
print(5<4)
print(5<=4)
print(4 == 3)
print(5 != 4)
print(1&1)
print(1&0)
print(1|1)
print(1|0)
print(True&True)
print(True&False)
print(True|True)
print(True|False)


# In[ ]:


# Variable assignments


# In[ ]:


x = 3
y = 6
print(x+y)
print(x/3)
print(x*y)
print(x%y)
print(x**y)# 
print(pow(x,y))


# In[ ]:


x,y=3,3
print(y == x)
print(x, y)
x, y = y, x
print(x, y)
x, y = y, x*y+y
print(x, y)
x, y, z = 1, '2', 'hello'
print(x, y, z)


# In[1]:


# in OPERATOR
x=[1,2,3,'a']
print(1 in x)
print (5 in x)


# In[ ]:



## Print formatting


# In[ ]:


x=2
y=3
print(x)
print("the variable x has value: ", x) #This gives a tuple in python 2.x
print('the variable x has value: {}'.format(x))
print('{}, {}'.format(x, y))
print('{1}, {0}'.format(x, y)) #positional formatting


# In[ ]:


## Loops and Conditional statements


# In[2]:


# how to use for
for i in [1, 2, 3, 4]:
    
    print(i)
# observe the difference between previous for and the below for
for i in [1, 2, 3, 4]:
    print (i), #the difference is a comma

print('\n')
#max of two numbers using simple if 
a,b=5,6
if a>b:
    print(a)
else:
    print(b)



# In[9]:


l = range(1,20,3)
for i in l:
    print(i)


# In[11]:


print('a'*2)
print('\n'*2)
print('b')


# In[ ]:


# usage of range is to create a sequence
range_list = range(5)
#above statement is equivalent to "list = [0, 1, 2, 3, 4]"
print(range_list)
print('\n'*2)
# usage of simple else if ladder
for i in range_list:
    if i < 2:
        print('{} < 2'.format(i))
    elif i == 2:
        print('{} = 2'.format(i))
    else:
        print('{} > 2'.format(i))
print('='*50)


# In[ ]:


### Question: Print all the natural numbers below 100 which are multiples of 3 or 5


# In[ ]:


# Your answer here
xlist = range(1,100)
for i in xlist:
    if i%3==0 or i%5==0:
        print('{} is a multiple of 3 or 5'.format(i))


# In[ ]:


### Write a program which will find all such numbers which are divisible by 11 but are not a multiple of 5, between 2000 and 3200 (both included). The numbers obtained should be printed in a comma-separated sequence on a single line.
Hint: Consider use range(#begin, #end) method


# In[ ]:


# Your Answer here
xlist = range(2000,3201)
ylist=[]
for i in xlist:
    if i%11==0 and i%5 != 0:
        print(i, sep = ", ", end = ' ', flush = True)
        #ylist.append(i),
#print(ylist)


# In[ ]:


## LISTS AND INDEXING


# In[12]:


list1 = ['Machinelearning', 'TextMIning', 2014, 2016,1.3]
list2 = [1, 2, 3, 4, 5 ];
list3 = ["a", "b", "c", "d"]
print(list1)
print(list2)
print(list3)


# In[13]:


# Accessing Values from List
list1[0:2]


# In[ ]:


#individual elements can be accessed through index
print(list1[0])# indexing starts from zero
print(list1[1])
print(list1[2])
print(list1[0:5])# : scope operate can enable you to access multiple elements till 5-1 position
print("length of list1",len(list1))## len gives legth (number of elements in list) of list
print(list1[0:len(list1)])
print(list2[0:6])
print(list2[0:len(list2)])
print(list3[0:len(list3)])

combined=list1+list2+list3
print(combined)
list1.extend(['a','b'])
print(list1)

list = [1, 2, 3, 11, 12, 13, 3, 4]
print(sum(list))


# In[ ]:


## Guess the out put of the following with out executing 


# In[ ]:


list = [1, 2, 3,'A', '12', 13, 3, 4,'hai']
print(list[0:4])
print(list[:6])
print(list[2:8])


# In[ ]:


print(list)
print(list[-1]) #prints the last element
print(list[-3])
print(list[3:-3])


# In[ ]:


# different methods in lists


# In[15]:


list = [1, 2, 3, 11, 12, 13, 3, 4]
print(list)
list.append(0)
print(list)
list.extend([12, 13, 14])
print(list)
list.index(1)


# In[ ]:


list = [1, 2, 3, 11, 12, 13, 3, 4]
print(list)
list.append(0)
print(list)
list.extend([12, 13, 14])
print(list)
print(list+[1,2,3,4])
print(len(list))
print(list.index(1))
list[1]=23# lists can be updated
print(list)

#just another example use of append
l=range(100)
print(l)
l1=[]
for i  in l :
    l1.append(i)
print("after copying")
print(l1)
type(l)


# In[19]:


# List Comprehensions


# In[ ]:


list = [1, 2, 3, 4, 5, 6]
print([i for i in list])
print([i*12+3 for i in list])
a = [i*12+3 for i in list]
print(a)
print([i*12 for i in list if i > 2])


# In[ ]:


### A String is a list too!!
###IT IS CONSIDERED AS A LIST OF CHARACTERS
###please observe the output of the following statements


# In[ ]:


string = 'Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam'
print(string)
print(string[0])
print(string[:100])
print(string.split())
print(string.split('e'))
print(string.lower())
print(string.replace('e', 'a'))
print(len(string))
print(string.count('sit'))
tokens = string.split()
print(tokens)
print(' '.join(tokens))
print(' abc '.join(tokens))


# In[ ]:


## Explore different methods in strings given para_str


# In[ ]:


# # para_str = """this test is a long string that is made up of
# # several lines and non-printable characters such as
# # TAB ( \t ) and they will show up that way when displayed.
# # NEWLINEs within the string,test  whether explicitly given like
# # this within the brackets [ \n ], or just a NEWLINE within
# # the variable assignment will also test show up.
# # """

# # ### yOUR CODE HERE
# # # Convert this string to lower case
# # print(para_str.lower())
# # # 
# # # Covert this string to upper case
# # print(para_str.upper())
# # # Only the first alphabet of this sentence should be capital
# # print(para_str.capitalize())
# # # Split this string into words and name the list as z.
# # z = para_str.split()
# # print(z)
# # # Replace the word ‘test’ with ‘tested’ in para_str 
# # para_str.replace('test','tested')
# # # Replace the word ‘tested’ with ‘test’ in the list para_str.
# # para_str.replace('tested','test')

# # # Now combine the words in the list and make the sentence.
# s=' '.join(z)
# print(s)

# # # In the new sentence find if there is a letter ‘z’ in it
# s.find('l')

# # # Replace spaces with hyphens in new sentence. Then make the words uppercase and split on hyphens in a single line
# s.replace(' ','-')
# # dir(s)

get_ipython().magic('pinfo string.splitlines')


# In[ ]:


## SETS


# In[ ]:



l1=['A','B','C','D','E']
l2=['A']
print(l1+l2)
print("operations on sets")
s1=set(l1)
s2=set(l2)
print(s1.issuperset(s2))#s1>=s2
print(s1.union(s2))#s1|s2
print(s1.intersection(s2))#s1&s2
print(s1.difference(s2))#s1-s2
#JUST CHECKING ALL OPERATIONS
print(s1>=s2)#
print(s1|s2)#s1|s2
print(s1&s2)#s1&s2
print(s1-s2)#s1-s2
# usage of 'in' operator with sets
print('A' in l1)
print('F' in l2)
#just some additional things 
# set comprehension
a = {x for x in 'abracadabra' if x not in 'abc'}
print(a)
b={x for x in s1 if x  in s2}
print(b)


# In[ ]:


## Tuples


# In[ ]:


t1=(1,2,3) # 1,2,3
print (t1)
type(t)
# tuples can be nested
t2=((1,2,3),(5,10,15)) # or (1,2,3),(5,10,15)
print(t2)
print(t2[0])
print(t2[1])
print(t2[0][1])
print(t2[0][2])
#t2[0]=4,5,6 # generates error tuples are immutable
#print(t2) # generates error tuples are immutable

# but tuple can contain mutable objects
t3=([1,2,3],['hai','python','2.7'],"welcome")
print(t3)
t3[0][0]=10# i am updating list present in tuple
print(t3)
# you can also assign a tuple to multiple variables 
x,y,z=t1
print("x is",x)
print(y)
print(z)
# usage of 'in' with tuples
print(1 in t1)
print(9 in t1)


# In[ ]:


## Dictionaries


# In[ ]:


my_dict = {'key1':'value1', 4.0:'4', '3':3.22, 'list':['a', 'b', 3], 'key2':'value2'}
print(my_dict['key1'])
print(my_dict['3'])
print(my_dict[4.0])
my_dict['3'] += 1
print(my_dict['3'])
# print(my_dict[3])
print(my_dict['list'])
print('keys --> {}'.format(my_dict.keys()))
print('values --> {}'.format(my_dict.values()))


# In[ ]:


c1 = {'word1': 1000, 'word2':299}
c2 = {'word2': 300, 'word1':30}

word_count = c1.copy()
word_count.update(c2)

print(word_count)


# In[ ]:


## Functions


# In[32]:


def mult(*):
    product = input1*input2
    return(product)

mult(3, 4,6,78)


# In[ ]:


### Question: Define a function max() that takes two numbers as arguments and returns the largest of them. 
##Use the if-then-else construct available in Python. (It is true that Python has the max() function built in, 
##but writing it yourself is nevertheless a good exercise.)


# In[33]:


def max(a,b):
    if(a>b):
        print('{} is max'.format(a))
    else:
        print('{}is max'.format(b))
max(5,6)

### Question: Define a function sum() and a function multiply() that sums and multiplies (respectively) all the numbers in a list of numbers. For example, sum([1, 2, 3, 4]) should return 10, and multiply([1, 2, 3, 4]) should return 24.
# In[ ]:


####WRITE YOUR ANSWER HERE

def sum1(numbers):
    x=0
    for i in numbers:
        x=x+i
    print(x)

sum1([1,2,1,2])

def product1(numbers):
    p=1
    for i in numbers:
        p=p*i
    print(p)
product1([1,1,3])


# In[ ]:


### Question: Print the first 100 elements of a Fibonacci series (1, 1, 2, 3, 5, 8, 13, 21, 34, ...)


# In[ ]:


##WRITE YOUR ANSWER HERE

## Modules

Modules are the backbone of python programs. Much like libraries in 'R', these need to be loaded before calling any useful function.
# In[38]:


#Importing a whole module and accessing any function in it is as follows
import math
print(math.pow(4, 3))

#Imporing a particular function from a module is as follows
from random import randint
print(randint(2, 39)) #Generates a random integer between 2 and 39

import random as rd
print(rd.randint(1,5))


# In[36]:


get_ipython().magic('pinfo math.pow')


# In[4]:


##NUMPY Arrays


# In[41]:


import numpy as np


# In[39]:


list1 = [2,5,9,8]
list2 = [4,10,18,16]


# In[40]:


list2/list1


# In[42]:


np_list1 = np.array(list1)
np_list2 = np.array(list2)


# In[43]:


np_list2/np_list1


# In[ ]:


#2d numpy array


# In[44]:


np_2d = np.array([[2, 5, 9, 8],
                  [4,10,18,16]])
np1_2d = np.array([list1,list2])


# In[45]:


np_2d[0]


# In[46]:


np_2d[0][2]


# In[47]:


np_2d[0,2]


# In[48]:


np1_2d[1,3]


# In[49]:


np_2d[:,2]


# In[50]:


np_2d[1,:]


# In[51]:


print(np.mean(np_2d[1,:]))
sum([4, 10, 18, 16])/4


# In[ ]:


np.median(np_2d[1,:])


# In[ ]:


np.std(np_2d[0,:])


# In[ ]:


np.corrcoef(np_2d[0,:],np_2d[0,:])


# In[52]:


#simulating random numbers
x = np.random.normal(2.4,0.1,100)
x[:5]


# In[53]:


y = np.round(x,2)


# In[54]:


y[:5]


# In[ ]:


#Pandas
#pandas is a python library that helps in creating and manipulating dataframes that contain rows and columns
#basic data structures in pandas are Series, Dataframe and Panel
#dataframes are made of series and panels are made of dataframes


# In[57]:


import pandas as pd


# In[59]:


#Series is a one-dimensional array like structure with homogeneous data. 
s = pd.Series([1,3,5,2,48,7])


# In[ ]:


print(s)


# In[ ]:


s[2]


# In[64]:


#A Data frame is a two-dimensional data structure with rows and columns. Similar to a dataframe in R and excel
#columns can be of different data types
#rows and columns have names/labels (indices in the case of rows)
df = pd.DataFrame()


# In[66]:


print(df)


# In[ ]:


#acutal syntax
pd.DataFrame( data, index, columns, dtype, copy)
#data can be either an np array, lists, series, dictionary etc
#index is row labels and columns is column names


# In[75]:


df = pd.DataFrame(list1)


# In[ ]:


print(df)


# In[ ]:


df = pd.DataFrame(list1,columns=['Numbers'])
print(df)


# In[ ]:


df = pd.DataFrame(list1,columns=['Numbers'],index = ['a','b','c','d'])
print(df)


# In[ ]:


#column selection
d = {'one' : pd.Series([1, 2, 3], index=['a', 'b', 'c']),
      'two' : pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])}

df = pd.DataFrame(d)
print (df ['one'])


# In[ ]:


#adding a new column
print(df)
df['three']=pd.Series([10,20,30],index=['a','b','c'])
print(df)


# In[ ]:


#row selection
print (df.loc['b'])


# In[ ]:


#row slicing
print (df[2:4])


# In[ ]:


#adding rows
df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
df = df.append(df2)
print(df)


# In[77]:


#head, tail, describe, info,dtypes,shape,size
df.info()

