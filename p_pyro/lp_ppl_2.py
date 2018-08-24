import picos as pic


QLP  = pic.Problem()


x = QLP.add_variable('x',1) 
y = QLP.add_variable('y',1) 


QLP.add_constraint( 1| x > 0 )
QLP.add_constraint( 1| y > 0 )
QLP.add_constraint( 1| x < 1 )
QLP.add_constraint( 1| y < 1 )
QLP.add_constraint( 1| x*y == 0.69 )



QLP.solve(verbose=0)
print QLP

print x
print y
