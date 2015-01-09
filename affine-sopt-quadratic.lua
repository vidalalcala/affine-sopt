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
   -N,--nbIter   (default 10)      number of iterations
]])

-- threads
-- torch.setnumthreads(opt.threads)
-- print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- Initial seed
torch.manualSeed(123)

-- Create synthetic data
local p = opt.nbPar
local n = p + 1      --required for full rank initialization
local N = opt.nbIter

local H = torch.rand(p, p)
H = ( H:t() + H )
H:mul(0.5)
local Sigma = torch.rand(p, p)
Sigma = ( Sigma:t() + Sigma )
Sigma:mul(0.5)
local alphaOptimal = torch.rand(1, p)

-- print problem setup
print('H : ')
print(H)
local HInverse = torch.inverse(H)
print('HInverse : ')
print(HInverse)
print('Sigma : ')
print(Sigma)
print('alphaOptimal : ')
print(alphaOptimal)

-- construct the stochastic gradient sampler
function  stochasticGradientQuadratic(alpha)
  local Z = torch.Tensor(1, p)
  randomkit.normal(Z, 0, 1)
  local estimator = alpha * H - (Z * Sigma) * H - alphaOptimal * H
  return estimator
end

-- generate first n = p + 1 states randomly
local X = torch.rand(n, p)

print(X)

-- Initialize the estimators
local Y = torch.Tensor(n, p)
for i = 1, n, 1 do
  local alpha = torch.Tensor(1, p)
  alpha:copy(X[{i,{}}])
  local gradAlpha = stochasticGradientQuadratic(alpha)
  Y[{i,{}}] = gradAlpha
end

local X = torch.cat(X , torch.ones(n, 1), 2)
local P = torch.inverse( X:t() * X )
local B = P * (X:t() * Y)
local G = torch.inverse(B[{{1,p},{}}])

-- 
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
  BAdd = (P * x:t()) * (y - x * B)
  BAdd:mul(alpha[1][1])
  B = B + BAdd
  PAdd = ( P * x:t() * x * P:t() )
  PAdd:mul(- alpha[1][1])
  P = P + PAdd
  b = ( v:t() * G * u + 1.0)
  local beta = torch.inverse(b)
  GAdd = (G * u * v:t() * G)
  GAdd:mul(beta[1][1])
  G = G + GAdd
end

--


