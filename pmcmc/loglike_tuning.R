## Tune number of particles to get Var(log likelihood) roughly equal to 1.5
library(pomp)

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

nparticles_seq = seq(from=10, to=200, by=10)
sd_est = 0 * nparticles_seq
time_est = sd_est
cat("\n")
for (i in 1:length(nparticles_seq)) {
    Np = nparticles_seq[i]
    cat("Running", Np, "particles\n")
    start_time = Sys.time()
    temp = replicate(100, pfilter(lorenz, Np=Np, tol=0)@loglik)
    end_time = Sys.time()
    time_est[i] = end_time - start_time
    temp = replicate(100, pfilter(lorenz, Np=Np, tol=0)@loglik)
    if (min(temp) == -Inf) {
        cat("Some particle filters failed\n")
        temp = temp[temp>-Inf]
    }
    sd_est[i] = sd(temp)
    cat("Standard deviation is", sd_est[i],"\n")
}

plot(nparticles_seq, sd_est, log="y")
abline(h=1.5)

plot(nparticles_seq, time_est)

## Now the case where sigma is fixed to 0.1
lorenz@params=c(th1=log(10), th2=log(28), th3=log(8/3), sigma=log(0.1))

nparticles_seq = c(1E3, 1E4, 1E5, 1E6)
sd_est = 0 * nparticles_seq
time_est = sd_est
cat("\n")
for (i in 1:length(nparticles_seq)) {
    Np = nparticles_seq[i]
    cat("Running", Np, "particles\n")
    start_time = Sys.time()
    temp = replicate(100, pfilter(lorenz, Np=Np, tol=0)@loglik)
    end_time = Sys.time()
    time_est[i] = end_time - start_time
    if (min(temp) == -Inf) {
        cat("Some particle filters failed\n")
        temp = temp[temp>-Inf]
    }
    sd_est[i] = sd(temp)
    cat("Standard deviation is", sd_est[i],"\n")
}

plot(nparticles_seq, sd_est, log="xy")
abline(h=1.5)

plot(nparticles_seq, time_est)
