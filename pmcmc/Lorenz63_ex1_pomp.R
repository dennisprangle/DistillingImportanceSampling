# To run from the shell use "R CMD BATCH --no-save Lorenz63_ex1_pomp.R"
library(pomp)
library(mcmcse)

# Data (as in Lorenz_data.npy)
yobs = c(-13.55, -16.19,  30.77,
          1.51,  -3.54,  14.27,
        -18.93, -21.09,  32.1,
         10.09,   9.88,  31.29,
          5.31,   6.76,  19.93)

yobs = matrix(yobs, ncol=3, byrow=TRUE)
colnames(yobs) = c("y1","y2","y3")
yobs = data.frame(yobs)

lorenz = pomp(data=yobs, times=0.4*1:5, t0=0,
  ## nb Large x1,x2,x3 values can become non-finite.
  ## Zero likelihood is appropriate for these.
  dmeasure=Csnippet("
                    if (!R_FINITE(x1) || !R_FINITE(x2) || !R_FINITE(x3)) {
                        lik = R_NegInf;
                    } else {
                        lik = dnorm(y1,x1,exp(sigma),1) +
                              dnorm(y2,x2,exp(sigma),1) +
                              dnorm(y3,x3,exp(sigma),1);
                    }
                    lik = (give_log) ? lik : exp(lik);
                    "),
  rprocess=discrete_time(
    step.fun=Csnippet("
                      double tx1, tx2, tx3;                      
                      tx1 = x1 + dt*exp(th1)*(x2-x1) + rnorm(0,sqrt(10*dt));
                      tx2 = x2 + dt*(exp(th2)*x1 - x2 - x1*x3) +
                            rnorm(0, sqrt(10*dt));
                      tx3 = x3 + dt*(x1*x2 - exp(th3)*x3) +
                            rnorm(0, sqrt(10*dt));
                      x1 = tx1; x2 = tx2; x3 = tx3;
                      "),
                      delta.t=0.02),
  rinit=Csnippet("
                 x1 = -30; x2 = 0; x3 = 30;
                 "),
  obsnames=c("y1","y2","y3"),
  statenames=c("x1","x2","x3"),
  paramnames=c("th1","th2","th3","sigma"),
  params=c(th1=log(10), th2=log(28), th3=log(8/3), sigma=log(2))
)

lorenz.dprior = function(object, params, ..., log=FALSE) {
  d = sum(dexp(exp(c(...)), rate=0.1, log=TRUE)) # parameters end up in ...
  if (log) d else exp(d)
}

prop.var = rbind(c( 0.016, 0.002,  0.003,-0.008),
                 c( 0.002, 0.001,  0.000, 0.000),
                 c( 0.003, 0.000,  0.007,-0.001),
                 c(-0.008, 0.000, -0.001, 0.075))

colnames(prop.var) = c("th1","th2","th3","sigma")
rownames(prop.var) = c("th1","th2","th3","sigma")

mcmc_out = pmcmc(lorenz, Nmcmc = 8E4, Np=50,
                 dprior = lorenz.dprior,
                 proposal = mvn.rw(prop.var),
                 verbose=FALSE)

out = data.frame(traces(mcmc_out))[,4:7]

multiESS(out)

write.csv(file="Lorenz63_ex1_mcmc.csv", out, row.names=FALSE)
