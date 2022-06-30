"""
-------------------------------------------------------------
        EE2703 Applied Programming Lab - 2022(Jan-May)
                        Assignment 2
        Name : Chagari Koushal Kumar Reddy
        Roll Number : EE20B023
        Date : 09-02-2022
-------------------------------------------------------------
"""

# Importing necessary modules
from sys import argv, exit
import numpy as np
import cmath

# Using constant variables instead of hard coding and also improving readability
CIRCUIT = '.circuit'
END = '.end'
pi = cmath.pi

class resistor:
    def __init__(self, name, node1, node2, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = metricPrefixes2Value(value)

class capacitor:
    def __init__(self, name, node1, node2, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = metricPrefixes2Value(value)

class inductor:
    def __init__(self, name, node1, node2, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = metricPrefixes2Value(value)

class Independent_Voltage_Source:
    def __init__(self, name, positiveNode, negativeNode, value):
        self.name = name
        self.positiveNode = positiveNode
        self.negativeNode = negativeNode
        self.value = metricPrefixes2Value(value)

class AC_Voltage_Source:
    def __init__(self, name, positiveNode, negativeNode, peak2peak, phase):
        self.name = name
        self.positiveNode = positiveNode
        self.negativeNode = negativeNode
        self.peak2peak = metricPrefixes2Value(peak2peak)
        self.phase = phase
        self.phasor = cmath.rect(float(peak2peak)/2,float(phase))

class Independent_Current_Source:
    def __init__(self, name, fromNode, toNode, value):
        self.name = name
        self.fromNode = fromNode
        self.toNode = toNode
        self.value = metricPrefixes2Value(value)

class AC_Current_Source:
    def __init__(self, name, fromNode, toNode, peak2peak, phase):
        self.name = name
        self.fromNode = fromNode
        self.toNode = toNode
        self.peak2peak = metricPrefixes2Value(peak2peak)
        self.phase = self.phase
        self.phasor = cmath.rect(float(peak2peak)/2, float(phase))

class VCVS:
    def __init__(self, name, positiveNode, negativeNode, controllingPositiveNode, controllingNegativeNode, value):
        self.name = name
        self.positiveNode = positiveNode
        self.negativeNode = negativeNode
        self.controllingPositiveNode = controllingPositiveNode
        self.controllingNegativeNode = controllingNegativeNode
        self.proportionalityFactor = metricPrefixes2Value(value)

class VCCS:
    def __init__(self, name, fromNode, toNode, controllingPositiveNode, controllingNegativeNode, value):
        self.name = name
        self.fromNode = fromNode
        self.toNode = toNode
        self.controllingPositiveNode = controllingPositiveNode
        self.controllingNegativeNode = controllingNegativeNode
        self.transConductance = metricPrefixes2Value(value)

class CCVS:
    def __init__(self, name, positiveNode, negativeNode, controllingCurrent_VS, value):
        self.name = name
        self.positiveNode = positiveNode
        self.negativeNode = negativeNode
        self.controllingCurrent_VS = controllingCurrent_VS
        self.transResistance = metricPrefixes2Value(value)

class CCCS:
    def __init__(self, name, fromNode, toNode, controllingCurrent_VS, value):
        self.name = name
        self.fromNode = fromNode
        self.toNode = toNode
        self.controllingCurrent_VS = controllingCurrent_VS
        self.proportionalityFactor = metricPrefixes2Value(value)

# Converts metric prefixes like kilo, milli, micro, nano, pico to their actual values
def metricPrefixes2Value(value):
    lastLetter = value[-1]
    if lastLetter == 'k' or lastLetter == 'm' or lastLetter == 'u' or lastLetter == 'n' or lastLetter == 'p':
        base = float(value[:-1])       # Everything other than the last prefix
        Converter = { 'k': 1e3, 'm' : 1e-3, 'u' : 1e-6, 'n' : 1e-9, 'p' : 1e-12 }
        return base*Converter[lastLetter]
    else:
        return float(value)

def spiceLine2Tokens(spiceLine):
    ''' Function analyzes the components, creates objects of that particular type,
        adds it to the element dictionary and helps to find the dimension of the nodal matrix
    '''
    spiceLine = spiceLine.strip()
    spiceLine = spiceLine.split()
    firstLetter = spiceLine[0][0]
    global element_dictionary
    global nodes_list
    global AuxilaryVariables_No

    # Resistor or Capacitor or Inductor
    if(firstLetter == 'R' or firstLetter == 'L' or firstLetter == 'C'):
        node1 = spiceLine[1]
        node2 = spiceLine[2]
        value = spiceLine[3]
        if(firstLetter == 'R'):
            elementName = "Resistor " + spiceLine[0]
            element_dictionary['Resistor'].append(resistor(elementName, node1, node2, value))
        if(firstLetter == 'L'):
            elementName = "Inductor " + spiceLine[0]
            element_dictionary['Inductor'].append(inductor(elementName, node1, node2, value))
        if(firstLetter == 'C'):
            elementName = "Capacitor " + spiceLine[0]
            element_dictionary['Capacitor'].append(capacitor(elementName, node1, node2, value))
        nodes_list.extend([node1, node2])
    
    # Independent DC Voltage Source
    elif(firstLetter == 'V' and spiceLine[3] == 'dc'):
        elementName = spiceLine[0]
        positiveTerminal = spiceLine[1]
        negativeTerminal = spiceLine[2]
        value = spiceLine[4]
        AuxilaryVariables_No += 1
        nodes_list.extend([positiveTerminal,negativeTerminal])
        element_dictionary['IVS'].append(Independent_Voltage_Source(elementName, positiveTerminal, negativeTerminal, value))
    
    # Independent AC Voltage Source
    elif(firstLetter == 'V' and spiceLine[3] == 'ac'):
        elementName = spiceLine[0]
        positiveNode = spiceLine[1]
        negativeNode = spiceLine[2]
        peak2peak = spiceLine[4]

        # Phase should be in radians
        phase = spiceLine[5]
        nodes_list.extend([positiveNode, negativeNode])
        AuxilaryVariables_No += 1
        element_dictionary['ACVS'].append(AC_Voltage_Source(elementName, positiveNode, negativeNode, peak2peak, phase))

    # Independent Current Source
    elif(firstLetter == 'I' and spiceLine[3] == 'dc'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]
        value = spiceLine[4]
        nodes_list.extend([fromNode, toNode])
        element_dictionary['ICS'].append(Independent_Current_Source(elementName, fromNode, toNode, value))
    
    # Independent AC Current Source
    elif(firstLetter == 'I' and spiceLine[3] == 'ac'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]
        peak2peak = spiceLine[4]
        phase = spiceLine[5]
        nodes_list.extend([fromNode, toNode])
        element_dictionary['ACCS'].append(AC_Current_Source(elementName, fromNode, toNode, peak2peak, phase))
    
    # VCVS
    elif(firstLetter == 'E'):
        elementName = spiceLine[0]
        PositiveNode = spiceLine[1]
        NegativeNode = spiceLine[2]
        controllingPositiveNode = spiceLine[3]
        controllingNegativeNode = spiceLine[4]
        proportionalityFactor = spiceLine[5]
        element_dictionary['VCVS'].append(VCVS(elementName, PositiveNode, NegativeNode, controllingPositiveNode, controllingNegativeNode, proportionalityFactor))
        nodes_list.extend([PositiveNode, NegativeNode, controllingPositiveNode, controllingNegativeNode])
        AuxilaryVariables_No += 1
    
    # CCCS
    elif(firstLetter == 'F'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]
        controllingCurrent = spiceLine[3]
        proportionalityFactor = spiceLine[4]
        element_dictionary['CCCS'].append(CCCS(elementName,fromNode, toNode, controllingCurrent, proportionalityFactor))
        nodes_list.extend([fromNode, toNode])

    # VCCS
    elif(firstLetter == 'G'):
        elementName = spiceLine[0]
        fromNode = spiceLine[1]
        toNode = spiceLine[2]      
        controllingPositiveNode = spiceLine[3]
        controllingNegativeNode = spiceLine[4]
        transConductance = spiceLine[5]
        element_dictionary['VCCS'].append(VCCS(elementName, fromNode, toNode, controllingPositiveNode, controllingNegativeNode, transConductance))
        nodes_list.extend([fromNode, toNode, controllingPositiveNode, controllingNegativeNode])

    # CCVS
    elif(firstLetter == 'H'):
        elementName = spiceLine[0]
        PositiveNode = spiceLine[1]
        NegativeNode = spiceLine[2]
        controllingCurrent = spiceLine[3]
        transResistance = spiceLine[4]
        element_dictionary['CCVS'].append(CCVS(elementName, PositiveNode, NegativeNode, controllingCurrent, transResistance))
        nodes_list.extend([PositiveNode,NegativeNode])
        AuxilaryVariables_No += 1


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

            # Introducing an ac_flag 
                ac_flag = 0
                if len(netlistLines)>(end+1) and netlistLines[end+1][:len('.ac')] == '.ac':
                    ac_flag = 1

            # Finding the frequency of AC operation
                if ac_flag == 1:
                    ac_line = netlistLines[end+1]
                    ac_line = ((ac_line.split('#')[0]).strip()).split()
                    freq = metricPrefixes2Value(ac_line[2])

            # Number of auxilary variables is equal to the number of independent/dependent voltage sources
                AuxilaryVariables_No = 0
                nodes_list = []
                nodes_dictionary = {}
                
            # Element dictionary to keep track of all classes of components
                element_dictionary = {"Resistor": [], "Capacitor": [], "Inductor" : [], "IVS" : [], "ACVS" : [], "ICS" : [], "ACCS": [], "VCVS": [], "CCCS" : [], "VCCS": [], "CCVS": []}
        
            # Analyzing every element present in the circuit using the function spiceLine2Tokens
                for line in netlistLines[start+1:end]:
                    spiceLine2Tokens((line.split('#')[0]).strip())

            # Making the entries in the nodal list unique
                nodes_list = list(set(nodes_list))

            # Dictionary of nodes: Assigning numbers to nodes using node name as key so that they correspond to rows/columns of nodal matrix   
                i = 1
                for node in nodes_list:
                    if node != '0' and node!='GND':
                        nodes_dictionary[node] = i
                        i += 1
                    else:
                        nodes_dictionary[node] = 0

                dimension_of_nodal_matrix = len(nodes_list) + AuxilaryVariables_No

                # G is the nodal matrix(square matrix)
                G = np.zeros((dimension_of_nodal_matrix, dimension_of_nodal_matrix), dtype = complex)
                Source = np.zeros((dimension_of_nodal_matrix,1), dtype = complex)

                '''
    The format of the variable matrix has the variables in the following order:
1. Nodal voltages
2. Currents through DC voltage souces
3. Currents through Controlled voltage sources(1st VCVS then CCVS)
4. Currents through AC voltage sources
    This ordering will be used below to construct the nodal matrix
                '''

            # Adding stamps of resistors to G
                for r in element_dictionary['Resistor']:
                    node1 = nodes_dictionary[r.node1]
                    node2 = nodes_dictionary[r.node2]
                    value = float(r.value)
                    G[node1][node1] += 1/(value)
                    G[node1][node2] += -1/(value)
                    G[node2][node1] += -1/(value)
                    G[node2][node2] += 1/(value)
                
            # Adding stamps of independent DC voltage sources to G      
                for v in element_dictionary['IVS']:
                    positiveNode = nodes_dictionary[v.positiveNode]
                    negativeNode = nodes_dictionary[v.negativeNode]
                    value = float(v.value)
                    aux_current_ind = len(nodes_list) + element_dictionary['IVS'].index(v)
                    G[positiveNode][aux_current_ind] += 1
                    G[negativeNode][aux_current_ind] += -1
                    G[aux_current_ind][positiveNode] += 1
                    G[aux_current_ind][negativeNode] += -1
                    Source[aux_current_ind][0] += value

            # Adding stamps of independent AC voltage sources to G
                for v in element_dictionary['ACVS']:
                    positiveNode = nodes_dictionary[v.positiveNode]
                    negativeNode = nodes_dictionary[v.negativeNode]
                    value = v.phasor
                    aux_current_ind = len(nodes_list) + len(element_dictionary['IVS']) + len(element_dictionary['VCVS']) + len(element_dictionary['CCVS']) + element_dictionary['ACVS'].index(v)
                    G[positiveNode][aux_current_ind] += 1
                    G[negativeNode][aux_current_ind] += -1
                    G[aux_current_ind][positiveNode] += 1
                    G[aux_current_ind][negativeNode] += -1
                    Source[aux_current_ind][0] += value
                
            # Adding stamps of capacitors to G
                for c in element_dictionary['Capacitor']:
                    node1 = nodes_dictionary[c.node1]
                    node2 = nodes_dictionary[c.node2]
                    capacitance = c.value
                    value = complex(0,-1/(2*pi*freq*capacitance))
                    G[node1][node1] += 1/(value)
                    G[node1][node2] += -1/(value)
                    G[node2][node1] += -1/(value)
                    G[node2][node2] += 1/(value)

            # Adding stamps of inductors to G
                for l in element_dictionary['Inductor']:
                    node1 = nodes_dictionary[l.node1]
                    node2 = nodes_dictionary[l.node2]
                    inductance = l.value
                    value = complex(0,2*pi*freq*inductance)
                    G[node1][node1] += 1/(value)
                    G[node1][node2] += -1/(value)
                    G[node2][node1] += -1/(value)
                    G[node2][node2] += 1/(value)

            # Adding stamps of independent current sources to G
                for ics in element_dictionary['ICS']:
                    fromNode = nodes_dictionary[ics.fromNode]
                    toNode = nodes_dictionary[ics.toNode]
                    value = ics.value
                    Source[fromNode][0] += -value
                    Source[toNode][0] += value

            # Adding stamps of AC current sources
                for accs in element_dictionary['ACCS']:
                    fromNode = nodes_dictionary[accs.fromNode]
                    toNode = nodes_dictionary[accs.toNode]
                    value = accs.phasor
                    Source[fromNode][0] += -value
                    Source[toNode][0] += -value
                
            # Adding stamps of Voltage Controlled Voltage Sources to G
                for vcvs in element_dictionary['VCVS']:
                    positiveNode = nodes_dictionary[vcvs.positiveNode]
                    negativeNode = nodes_dictionary[vcvs.negativeNode]
                    controllingPositiveNode = nodes_dictionary[vcvs.controllingPositiveNode]
                    controllingNegativeNode = nodes_dictionary[vcvs.controllingNegativeNode]
                    k = vcvs.proportionalityFactor
                    aux_current_ind = len(nodes_list) + len(element_dictionary['IVS']) + element_dictionary['VCVS'].index(vcvs)
                    G[positiveNode][aux_current_ind] += 1
                    G[negativeNode][aux_current_ind] += -1
                    G[aux_current_ind][controllingPositiveNode] += -k
                    G[aux_current_ind][controllingNegativeNode] += k
                    G[aux_current_ind][positiveNode] += 1
                    G[aux_current_ind][negativeNode] += -1

            # Adding stamps of Current Controlled Current Sources to G
                for cccs in element_dictionary['CCCS']:
                    fromNode = nodes_dictionary[cccs.fromNode]
                    toNode = nodes_dictionary[cccs.toNode]
                    controllingCurrent = cccs.controllingCurrent_VS
                    k = cccs.proportionalityFactor
            
        # Finding index of current variable in matrix. Controlling current can be thorugh IVS, ACVS
                    ind = 0
                    for j in range(len(element_dictionary['IVS'])):
                        v = element_dictionary['IVS'][j]
                        if v.name == controllingCurrent:
                            ind = len(nodes_list) + j
                    for j in range(len(element_dictionary['ACVS'])):
                        v = element_dictionary['ACVS'][j]
                        if v.name == controllingCurrent:
                            ind = len(nodes_list) + len(element_dictionary['IVS']) + len(element_dictionary['VCVS']) + len(element_dictionary['CCVS']) + j
                    
                    G[fromNode][ind] += k
                    G[toNode][ind] += -k

            # Adding stamps of Current Controlled Voltage Sources to G  
                for ccvs in element_dictionary['CCVS']:
                    positiveNode = nodes_dictionary[ccvs.positiveNode]
                    negativeNode = nodes_dictionary[ccvs.negativeNode]
                    k = ccvs.transResistance
                    controllingCurrent = ccvs.controllingCurrent_VS
            
            # Finding index of current variable in matrix. Controlling current can be thorugh IVS, VCVS, CCVS, ACVS
                    ind = 0
                    for j in range(len(element_dictionary['IVS'])):
                        v = element_dictionary['IVS'][j]
                        if v.name == controllingCurrent:
                            ind = len(nodes_list) + j
                    for j in range(len(element_dictionary['ACVS'])):
                        v = element_dictionary['ACVS'][j]
                        if v.name == controllingCurrent:
                            ind = len(nodes_list) + len(element_dictionary['IVS']) + len(element_dictionary['VCVS']) + len(element_dictionary['CCVS']) + j
                    
                    aux_current_ind = len(nodes_list) + len(element_dictionary['IVS']) + len(element_dictionary['VCVS']) + element_dictionary['CCVS'].index(ccvs)
                    
                    G[positiveNode][aux_current_ind] += 1
                    G[negativeNode][aux_current_ind] += -1
                    G[aux_current_ind][positiveNode] += 1
                    G[aux_current_ind][negativeNode] += -1
                    G[aux_current_ind][ind] += -k
            
            # Adding stamps of Voltage Controlled Current Sources to G
                for vccs in element_dictionary['VCCS']:
                    controllingPositiveNode = nodes_dictionary[vccs.controllingPositiveNode]
                    controllingNegativeNode = nodes_dictionary[vccs.controllingNegativeNode]
                    fromNode = nodes_dictionary[vccs.fromNode]
                    toNode = nodes_dictionary[vccs.toNode]
                    k = vccs.transConductance
                    G[fromNode][controllingPositiveNode] += k
                    G[fromNode][controllingNegativeNode] += -k
                    G[toNode][controllingPositiveNode] += -k
                    G[toNode][controllingNegativeNode] += k

            # We need to delete the following because in the actual nodal matrix, node 0/GND shouldn't be present
                # Delete 1st row in G
                G = np.delete(G,0,0)
                # Delete 1st column in G
                G = np.delete(G,0,1)
                # Delete 1st row in source matrix
                Source = np.delete(Source,0,0)

            # Printing the solution - nodal voltages and currents through voltage sources
                try:
                    solution = np.linalg.solve(G,Source)

                    # Printing node voltages
                    for node in nodes_list:
                        if node!= 'GND' and node!= '0':
                            print("The voltage at node %s is " %(node), end = '')
                            if(ac_flag == 0):
                                print(solution[nodes_dictionary[node] - 1][0].real, 'V')
                            else:
                                v = solution[nodes_dictionary[node] - 1][0]
                                print(v, 'V')
                                print("Magnitude:", abs(v), " Phase(in degrees):", cmath.phase(v)*180/pi, end = '\n\n')

                    # Printing currents through DC voltage sources               
                    for j in range(0,len(element_dictionary['IVS'])):
                        print("The current entering the positive terminal of the voltage source %s is " %(element_dictionary['IVS'][j].name), end = '')
                        if(ac_flag == 0):
                            print(solution[len(nodes_list)-1+j][0].real, "A")
                        else:
                            current = solution[len(nodes_list)-1+j][0]
                            print(current, "A")
                            print("Magnitude:", abs(current), " Phase(in degrees):", cmath.phase(current)*180/pi, end = '\n\n')
            
                    # Printing currents through AC voltage sources                     
                    for j in range(0,len(element_dictionary['ACVS'])):
                        print("The current entering the positive terminal of the voltage source %s is " %(element_dictionary['ACVS'][j].name), end = '')
                        current = solution[len(nodes_list) - 1 + len(element_dictionary['IVS']) + len(element_dictionary['VCVS']) + len(element_dictionary['CCVS']) + j][0]
                        print(current, "A")
                        print("Magnitude:", abs(current), " Phase(in degrees):", cmath.phase(current)*180/pi, end = '\n\n')
                    
                    # Printing currents through VCVS voltage sources
                    for j in range(len(element_dictionary['VCVS'])):
                        print("The current entering the positive terminal of the voltage source %s is " %(element_dictionary['VCVS'][j].name), end = '')
                        current = solution[len(nodes_list) - 1 + len(element_dictionary['IVS']) + j][0]            
                        if(ac_flag == 0):
                            print(current.real, "A")
                        else:
                            print(current, "A")
                            print("Magnitude:", abs(current), "Phase(in degrees):", cmath.phase(current)*180/pi)
                    
                    # Printing currents through CCVS voltage sources
                    for j in range(len(element_dictionary['CCVS'])):
                        print("The current entering the positive terminal of the voltage source %s is " %(element_dictionary['CCVS'][j].name), end = '')
                        current = solution[len(nodes_list) - 1 + len(element_dictionary['IVS']) + len(element_dictionary['VCVS']) + j][0]
                        if(ac_flag == 0):
                            print(current.real, "A")
                        else:
                            print(current, "A")
                            print("Magnitude:", abs(current), "Phase(in degrees):", cmath.phase(current)*180/pi)
                
                # Occurs when the G matrix is singular      
                except np.linalg.LinAlgError:
                    print("The given circuit is not solvable! Make sure that KCL and KVL can be obeyed at every node.")
                    print("This may happen because of loops containing only voltage sources, nodes connected to only current sources")

        # If the given file doesn't exist, then it is handled using this except block
        except IOError:
            print('Invalid file name! Please check if the given file exists in the directory')
            exit()