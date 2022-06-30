"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 1
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date : 26-01-2022
-------------------------------------------------------------
"""

# Importing necessary modules
from sys import argv, exit

# Using constant variables instead of hard coding and also improving readability
CIRCUIT = '.circuit'
END = '.end'

def spiceLine2Tokens(spiceLine):
    ''' Function analyzes and prints the element type, nodes connected to it and the relevant values 
        by taking respective line of netlist file as input
    '''
    spiceLine = spiceLine.strip()
    spiceLine = spiceLine.split()
    firstLetter = spiceLine[0][0]

    # Resistor or Capacitor or Inductor
    if(firstLetter == 'R' or firstLetter == 'L' or firstLetter == 'C'):
        if(firstLetter == 'R'):
            elementName = "Resistor " + spiceLine[0]
        if(firstLetter == 'L'):
            elementName = "Inductor " + spiceLine[0]
        if(firstLetter == 'C'):
            elementName = "Capacitor " + spiceLine[0]
        node1 = spiceLine[1]
        node2 = spiceLine[2]
        value = spiceLine[3]
        print("%s \nNodes: %s, %s \nValue: %s \n" %(elementName,node1,node2,value))
    
    # Independent Voltage Source
    elif(firstLetter == 'V'):
        elementName = spiceLine[0]
        positiveTermial = spiceLine[1]
        negativeTerminal = spiceLine[2]
        value = spiceLine[3]
        print("Independent Voltage source %s \nPositive Node: %s" %(elementName,positiveTermial), end = '  ')
        print("Negative Node: %s \nValue: %sV \n" %(negativeTerminal,value))
    
    # Independent Current Source
    elif(firstLetter == 'I'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]
        value = spiceLine[3]
        print("Independent Current Source %s \n To Node: %s" %(elementName,toNode), end = '  ')
        print("From Node: %s \n Value: %sA \n" %(fromNode,value))
    
    # VCVS
    elif(firstLetter == 'E'):
        elementName = spiceLine[0]
        PositiveNode = spiceLine[1]
        NegativeNode = spiceLine[2]
        controllingPostiveNode = spiceLine[3]
        controllingNegativeNode = spiceLine[4]
        proportionalityFactor = spiceLine[5]
        print("Voltage Controlled Voltage Source %s \n" %elementName)
        print("Proportionality Factor: %s  Controlling Positive Node: %s  Controlling Negative Node: %s" %(proportionalityFactor,controllingPostiveNode,controllingNegativeNode))
        print("Positive Node : %s  Negative Node: %s" %(PositiveNode,NegativeNode))
    
    # CCCS
    elif(firstLetter == 'F'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]
        controllingCurrent = spiceLine[3]
        proportionalityFactor = spiceLine[4]
        print("Current Controlled Current Source %s \nControlling current is through Voltage source: %s" %(elementName,controllingCurrent))
        print("Proportionality Factor: %s From Node: %s  To Node: %s" %(proportionalityFactor,fromNode,toNode))

    # VCCS
    elif(firstLetter == 'G'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]      
        controllingPostiveNode = spiceLine[3]
        controllingNegativeNode = spiceLine[4]
        transConductance = spiceLine[5]
        print("Voltage Controlled Current Source %s \n" %elementName)
        print("Transconductance: %s  Controlling Positive Node: %s  Controlling Negative Node: %s" %(transConductance,controllingPostiveNode,controllingNegativeNode))
        print("From Node: %s  To Node: %s" %(fromNode,toNode))

    # CCVS
    elif(firstLetter == 'H'):
        elementName = spiceLine[0]
        PositiveNode = spiceLine[1]
        NegativeNode = spiceLine[2]
        controllingCurrent = spiceLine[3]
        transResistance = spiceLine[4]
        print("Current Controlled Voltage Source %s \nControlling current is through Voltage source: %s" %(elementName,controllingCurrent))
        print("Transresistance: %s  Positive Node: %s  Negative Node: %s" %(transResistance,PositiveNode,NegativeNode))


# Checking the number of command line arguments given by the user
if len(argv) != 2:
    print('\n Invalid number of arguments! Follow the usage: %s <Netlistfile>' % argv[0])
    exit()
else:
    netlistFile = argv[1]

    # Checking if the given file is a netlist file
    if(not netlistFile.endswith(".netlist")):
        print("Invalid file type! Please give .netlist file")
    else:
        try:
            with open(argv[1]) as f:
                netlistLines = f.readlines()
                start = -1; end = -2

            # Extracting circuit definition start and end netlistLines
                for line in netlistLines:             
                    if CIRCUIT == line[:len(CIRCUIT)]:
                        start = netlistLines.index(line)
                    elif END == line[:len(END)]:
                        end = netlistLines.index(line)

            # Validating circuit block
                if start >= end or start == -1 or end == -2:         
                    print('Invalid circuit definition! Make sure you have .circuit and .end lines')
                    exit(0)

            # Printing the spice lines in reverse order
                for line in netlistLines[end-1:start:-1]:
                    line = line.split('#')[0]
                    line = (line.strip()).split()
                    line.reverse()
                    line = [' '.join(line)]
                    line = str(line[0])
                    print(line)

            # Printing every element present in the circuit using the function spiceLine2Tokens
                print("\n\nElements in the circuit in the order as present in the netlist are: ")
                for line in netlistLines[start+1:end]:
                    spiceLine2Tokens((line.split('#')[0]).strip())

        # If the given file doesn't exist, then it is handled using this except block
        except IOError:
            print('Invalid file name! Please check if the given file exists in the directory')
            exit()