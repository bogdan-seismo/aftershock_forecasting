import AftFore as aft

if __name__ == '__main__':
    
    t_learn = [0.0, 0.125]            # The range of the learning period [day].
    t_test  = [0.125, 0.250]            # The range of the testing period [day].
    Data    = './AftFore/gorj.dat'  # The path of the date file
    
    aft.EstFore(Data,t_learn,t_test)
