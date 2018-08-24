import picos as pic


QLP  = pic.Problem()


p1 = QLP.add_variable('p1',1) 
p2 = QLP.add_variable('p2',1) 
p3 = QLP.add_variable('p3',1) 
p12 = QLP.add_variable('p12',1) 
p13 = QLP.add_variable('p13',1) 
p23 = QLP.add_variable('p23',1) 

QLP.add_constraint( 1| p1 >= 0 ) # 1
QLP.add_constraint( 1| p2 >= 0 )
QLP.add_constraint( 1| p3 >= 0 )

QLP.add_constraint( 1| p1 >= p12 ) # 2
QLP.add_constraint( 1| p1 >= p13 )
QLP.add_constraint( 1| p2 >= p23 )

QLP.add_constraint( 1| p2 >= p12 )
QLP.add_constraint( 1| p3 >= p13 )
QLP.add_constraint( 1| p3 >= p23 )

QLP.add_constraint( 1| p1+p2+p3-p12-p13-p23 <= 1 )
QLP.add_constraint( 1| p1-p12-p13+p23 >= 0 )
QLP.add_constraint( 1| p2-p23-p12+p13 >= 0 )
QLP.add_constraint( 1| p3-p13-p23+p12 >= 0 )


QLP.solve(verbose=0)
print QLP

print(("p1 =  %s ; p2 =  %s ; p3 =  %s ; p12 =  %s ; p13 =  %s ; p23 = %s") % (p1,p2,p3,p12,p13,p23))