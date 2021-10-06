First_list = []
Second_list = []

def main():
# Init main inputs and result
a = int(input("Type first value: "))
b = int(input("Type second value: "))
result = 0

# Init a list that will contain halved values and one for double values
halved_values = []
double_values = []
# Create a list with halved values
def f_halve(x):
    halved_values.append(x) # append the input as first value
    while x > 1:
        x = int(x / 2) # remove decimals using int()
        halved_values.append(x)
    print("List halved values:",  halved_values)  # check first list is ok

# Double the second number (b) the same amount that time the first value (i.e. a) was halved
def f_doubling(x):
    if len(double_values) != len(halved_values):
        double_values.append(x)
        x = x * 2
        f_doubling(x) # recursion

# Create a list with halve values (without modifying a)
f_halve(a)
    
# Create a list with double values using a recursive function
f_doubling(b) 
print("List double values:", double_values) # check second list is ok
     
# Return a dictionary with couples like halved:double values
russian_molt = dict(zip(halved_values double_values))

# Sum double values associated with odd halved values
for value_a in russian_molt.keys():
    if value_a % 2 != 0:
        result = result + (russian_molt[value_a])

print("Result":  a x b  =  result)
        
    ########
    # MAIN #
    ########
    
    if __name__ == '__main__':
main()