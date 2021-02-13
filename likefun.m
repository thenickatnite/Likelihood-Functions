function ll = likefun(THETA, Y, X)
% multivariate normal likelihood function

est = vecparm(THETA);

Yhat = X*est.b;

like = mvnpdf(Y, Yhat, est.cov);

ll = sum(log(like));