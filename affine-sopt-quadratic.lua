-----------
-- Stochastic optimization using ordinary least squares to approximate the
-- Hessian. The code tests a variant of the sequential ordinary least squares
-- estimator developed -- by Alcala & Goodman with a noise quadratic loss 
-- function described by LeCun, Zhang and Schaul (2013) "No More Pesky Learning
-- Rates." .


require 'torch'
require 'pl'
require 'cutorch'

-- Use CUDA
-- print( cutorch.getDeviceProperties(cutorch.getDevice()) )
---torch.setdefaulttensortype('torch.CudaTensor')

-- Parse command-line options
local opt = lapp([[
   -t,--threads  (default 7)         number of threads
   -p,--nbPar    (default 4) 	     number of parameters
   -N,--nbIter   (default 10)      number of iterations
]])

-- threads
-- torch.setnumthreads(opt.threads)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- Initial seed
torch.manualSeed(123)

-- Create synthetic data
local p = opt.nbPar
local N = opt.nbIter

local X = torch.rand(N, p)
local H = torch.rand(p, p)
H = ( H:t() + H )
H:mul(0.5)
local Sigma = torch.rand(p, p)
Sigma = ( Sigma:t() + Sigma )
Sigma:mul(0.5)
local theta = torch.rand(p)
local Y = torch.Tensor(N, p)
local Z = torch.randn(N,p)

-- generate synthetis responses
Y = X * H - Z * Sigma * H
for i = 1, N, 1 do
  Y:select(1, i):addmv(H, theta)
end

-- Initialize the OLS estimator with p+1 observations.
sgdolsState = sgdolsState or {
            n = p + 1,
            parametersMean = torch.Tensor(nbPar):zero(),
            evalCounter = 0
            }
X = torch.cat(X, torch.ones(N, 1), 2) 
local P = torch.inverse(X:t() * X)
local B = P * (X:t() * Y)
local G = torch.inverse(B[{{1,p},{}}])
print('B : ')
print(B)
print('G : ')
print(G)

function SeqOls.addObservation(y, x, state)

end




-- 

--[[
int main (int argc, char * const argv[])
{
    //Initialize the random number generator
	mt19937 generator;
    normal_distribution<double> normalSample(0.0,1.0);
    
    // Create synthetic data
    int p = 4 ;                         // # of parameters
    int n = 1000000 ;                     // # of samples
    mat X = randu<mat>(n,p) ;           // The matrix of predictors
    mat H = randu<mat>(p,p) ;           // The Hessian matrix for the quadratic loss
    H = 0.5 * ( H.t() + H ) ;           // Hessian is symmetric
    mat Sigma = randu<mat>(p,p) ;       // The covariance matrix for the quadratic loss
    Sigma = 0.5 * ( Sigma.t() + Sigma ) ;// Covariance is symmetric
    mat theta = randu<mat>(1,p) ;       // The parameter we will try to estimate
    mat Y(n,p) ;                        // The matrix of responses
    mat Z(n,p) ;                        // Matrix with standard normal entries
    
    // Create a n x p matrix with standard normal entries
    for (int i = 0 ; i < n ; i++){
            for (int j = 0 ; j < p ; j++){
                Z(i,j) = normalSample(generator) ;
            }
    }
    
    //Generate synthetic responses
    Y = X * H - Z * Sigma * H ;
    for ( int i = 0 ; i < n ; i++){
        Y.row(i) = Y.row(i) + theta * H  ;
    }
    H.print("H: ") ;
    
    // Initialize the OLS estimator with p+1 observations.
    SeqOls Estimator ;
    Estimator.useObservations( X.rows(0,p) , Y .rows(0,p) ) ;
    Estimator.printEstimator() ;
    
    // Add the rest of the observations sequentially.
    for  ( int i = p + 1 ; i < n ; i++){
        Estimator.addObservation( X.row(i) , Y.row(i) ) ;
    }
    
    //Print the estimator
    Estimator.printEstimator() ;
    Estimator.testInverse();
}
]]
