""" 
This file is used to print the difference between the real value and the value predicted
"""



def affichage(tab):
    """
    ! affichage 
    print the value of the parameter as value1 ---> value 2 : value c
    @param tab  an array of tuple of 3 elements
    @return none
    """ 

    print("\n Tableau de comparaison")
    print("________________________________")
    for (a,b,c) in tab: 
        print("|   "+a+"  --->  "+b+"      : "+c)
    print("________________________________")
