import nelder_mead as nm 
import pso

choice = -1
string= "Select an option:\n\
        -----------------\n\
        1) Nelder Mead and Particle Swarm optimization\n\
        2) Nelder Mead optimization \n\
        3) Particle swarm optimization \n\
        0) Exit \n"

while(choice != 0):
    choice = int(input(string))
    
    if(choice==1):
        nm.main()
        pso.main()
    elif(choice==2):
        nm.main()
    elif(choice==3):
        pso.main()
    elif(choice != 0):
        print("invalid command")

print("Goodbye!")
