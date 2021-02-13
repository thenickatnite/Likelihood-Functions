function ll = likefun(THETA, Y, X)
% multivariate normal likelihood function

est = vecparm(THETA);

Yhat = X*est.b;

like = mvnpdf(Y, Yhat, est.cov); % est.cov comes from using the VECPARM function also included in this repository.

ll = sum(log(like));
