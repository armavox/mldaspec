
# coding: utf-8

# In[1]:


import sqlite3
conn = sqlite3.connect('employee.db')
# conn = sqlite3.connect(':memory:')  # Use RAM temp database
conn
c = conn.cursor()


# In[65]:


# c.execute("""CREATE TABLE employees (
#             first text,
#             last text,
#             pay integer
#             )""")


# In[12]:


c.execute(f"""INSERT INTO employees
            VALUES ('{emp_1.first}', '{emp_1.last}', {emp_1.pay})""")


# In[13]:


c.execute(f"""INSERT INTO employees
            VALUES (?, ?, ?)""", (emp_1.first, emp_1.last, emp_1.pay))


# In[28]:


c.execute(f"""INSERT INTO employees
            VALUES (:first, :last, :pay)""", {'first': emp_1.first, 'last': emp_1.last, 'pay': emp_1.pay})


# In[10]:


c.execute("""INSERT INTO employees 
            VALUES ('Tema', 'Lobantsev', 8000)""")


# In[7]:


c.execute("""SELECT * FROM employees 
            WHERE pay>10""")


# In[14]:


c.fetchone()


# In[8]:


c.fetchall()

c.fetchmany()
# In[11]:


conn.commit()


# In[31]:


conn.close()


# ###### Add python functionality

# In[2]:


from employee import Employee

emp_1 = Employee('John', 'Doe', 80000)
emp_2 = Employee('John', 'Smith', 90000)

import snippets

snippets.insert_emp(emp_2, conn)

snippets.get_emps_by_name('Smith', conn)

