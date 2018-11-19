
def insert_emp(emp, conn):
    with conn:
        c = conn.cursor()
        c.execute(f"""INSERT INTO employees
                    VALUES ('{emp.first}', '{emp.last}', {emp.pay})""")


def get_emps_by_name(lastname, conn):
    c = conn.cursor()
    c.execute("""SELECT * FROM employees 
                WHERE last=:last""", {'last': lastname})
    return c.fetchall()


def update_pay(emp, pay, conn):
    with conn:
        c = conn.cursor()
        c.execute("""UPDATE employees SET pay = :pay
                    WHERE first = :first AND last = :last""",
                  {'first': emp.first, 'last': emp.last, 'pay': pay})


def remove_emp(emp, conn):
    with conn:
        c = conn.cursor()
        c.execute("DELETE FROM employees WHERE first = :first AND last = :last",
                  {'first': emp.first, 'last': emp.last})
