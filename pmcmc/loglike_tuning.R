## Tune number of particles to get Var(log likelihood) roughly equal to 1.5
library(pomp)

# Data (as in Lorenz_data.npy)
yobs = c(-11.4800,  -4.1700,  36.7000,
         -10.8600, -17.5000,  22.4300,
           4.0900,   4.4300,  20.6200,
           6.1500,  13.3500,  12.5800,
          -1.4500,  -5.8100,  25.1700)

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
  # params are true values except for th1, where a value closer to the posterior mean is used
  params=c(th1=log(8), th2=log(28), th3=log(8/3), sigma=log(2))
)

nparticles_seq = seq(from=50, to=1000, by=50)
sd_est = 0 * nparticles_seq
time_est = sd_est
cat("\n")
for (i in 1:length(nparticles_seq)) {
    Np = nparticles_seq[i]
    cat("Running", Np, "particles\n")
    start_time = Sys.time()
    temp = replicate(100, pfilter(lorenz, Np=Np)@loglik)
    end_time = Sys.time()
    time_est[i] = end_time - start_time
    temp = replicate(100, pfilter(lorenz, Np=Np)@loglik)
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

## Now the case where sigma is fixed to 0.2
lorenz@params=c(th1=log(8), th2=log(28), th3=log(8/3), sigma=log(0.2))

nparticles_seq = c(1E3, 1E4, 1E5, 1E6)
sd_est = 0 * nparticles_seq
time_est = sd_est
cat("\n")
for (i in 1:length(nparticles_seq)) {
    Np = nparticles_seq[i]
    cat("Running", Np, "particles\n")
    start_time = Sys.time()
    temp = replicate(100, pfilter(lorenz, Np=Np)@loglik)
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

plot(nparticles_seq, time_est, log="x")
