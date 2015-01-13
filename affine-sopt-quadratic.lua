-----------
-- Stochastic optimization using ordinary least squares to approximate the
-- Hessian. The code tests a variant of the sequential ordinary least squares
-- estimator developed -- by Alcala & Goodman with a noise quadratic loss 
-- function described by LeCun, Zhang and Schaul (2013) "No More Pesky Learning
-- Rates." .

require 'pl'
require 'cutorch'
require 'randomkit'

-- Use CUDA
-- print( cutorch.getDeviceProperties(cutorch.getDevice()) )
---torch.setdefaulttensortype('torch.CudaTensor')

-- Parse command-line options
local opt = lapp([[
   -t,--threads  (default 7)         number of threads
   -p,--nbPar    (default 4) 	     number of parameters
   -N,--nbIter   (default 1000)        number of iterations
   -g,--gamma    (default 0.6)       power lag
]])

-- threads
-- torch.setnumthreads(opt.threads)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- Initial seed
torch.manualSeed(234)

-- Create synthetic data
local p = opt.nbPar
local nInit = p + 1      --required for full rank initialization
local N = opt.nbIter
local gamma = opt.gamma

local H = torch.rand(p, p)
H = H:t() * H 
local Sigma = torch.rand(p, p)
Sigma = Sigma:t() * Sigma
local alphaOptimal = torch.rand(1, p)

-- print problem setup
print('H : ')
print(H)
--print(torch.symeig(H))
local HInverse = torch.inverse(H)
print('HInverse : ')
print(HInverse)
print('Sigma : ')
print(Sigma)
--print(torch.symeig(Sigma))
print('alphaOptimal : ')
print(alphaOptimal)

-- construct the stochastic gradient sampler
function  stochasticGradientQuadratic(alpha)
  local Z = torch.Tensor(1, p)
  randomkit.normal(Z, 0, 1)
  local estimator = (alpha - alphaOptimal) * H - (Z * Sigma) * H 
  return estimator
end

-- generate first nInit = p + 1 states randomly
local X = torch.rand(nInit, p)
--print('X : ')
--print(X)

-- generate first nInit = p + 1 gradient observations
local Y = torch.Tensor(nInit, p)
for i = 1, nInit, 1 do
  local alpha = torch.Tensor(1, p)
  alpha:copy(X[{i,{}}])
  local gradAlpha = stochasticGradientQuadratic(alpha)
  Y[{i,{}}] = gradAlpha
end

--print('Y : ')
--print(Y)

-- Use first n = p +1 observations to construct initial estimators
local n = nInit
local X = torch.cat(X , torch.ones(n, 1), 2)
local P = torch.inverse( X:t() * X )
local B = P * (X:t() * Y)
local G = torch.inverse(B[{{1,p},{}}])
local HEstimator = G + G:t()
HEstimator:mul(0.5)

-- Online addition of observation
function addObservation(x, y)
  n = n + 1
  x = torch.cat(x, torch.ones(1,1), 2)
  local b = (x * P * x:t() + 1.0)
  local alpha = torch.inverse(b)
  local u = P * x:t()
  u = u[{{1,p},{}}]
  u:mul(alpha[1][1])
  local v = y - x * B
  v = v:t()
  BAdd = P * x:t() * (y - x * B)
  BAdd:mul(alpha[1][1])
  B = B + BAdd
  PAdd =  P * x:t() * x * P:t() 
  PAdd:mul(- alpha[1][1])
  P = P + PAdd
  b = ( v:t() * G * u + 1.0)
  local beta = torch.inverse(b)
  GAdd = (G * u * v:t() * G)
  GAdd:mul(-beta[1][1])
  G = G + GAdd
  HEstimator = G + G:t()
  HEstimator:mul(0.5)
end


-- Initial estimator
--print('initial H inverse estimator : ')
--print(HEstimator)
-- Initial estimator
--print('initial B estimator : ')
--print(B)

-- Perform Affine Invariant Stochastic Optimization
local alpha = torch.rand(1, p)
local alphaNew = torch.Tensor(1, p)
local gradAlpha = stochasticGradientQuadratic(alpha)
local gradAlphaNew = torch.rand(1, p)

for n = nInit + 1, N, 1 do
  alphaNew = gradAlpha*HEstimator
  alphaNew:mul(-1.0/(n^gamma))
  alphaNew = alphaNew + alpha
  gradAlphaNew = stochasticGradientQuadratic(alphaNew)
  addObservation(alpha, gradAlpha)
  alpha = alphaNew:clone()
  gradAlpha = gradAlphaNew:clone()
end

print('alpha estimator : ')
print(alphaNew)
print('H inverse estimator : ')
print(HEstimator)
print('gamma : ')
print(gamma)
print('number of gradient samples : ')
print(n)






