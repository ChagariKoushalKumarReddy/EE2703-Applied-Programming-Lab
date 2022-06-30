from scipy.integrate import quad

sq = lambda x : x*x

print((quad(sq,0,2))[0])