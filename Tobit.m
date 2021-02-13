function estMLE = Tobit(Y,X)
Yind = bsxfun(@eq, Y, 0);
% creates a vector of 1s and 0s that tell us whether Y is greater than 0
nx = size(X,2);
OLS_coef = (X'*X) \ X'*Y;

sv = [OLS_coef; log(std(Y))];

function est = vecparm(x)
    est = struct();
    est.b = x(1:nx);
    est.sigma = exp(x((nx+1):end));
    est.sigma2 = est.sigma^2;
end

function ll = calclike(x)
    est = vecparm(x);
    yhat = X*est.b;
    sigma = est.sigma;
    ll = sum(log(Yind .*(1-normcdf(yhat/sigma)) + (1-Yind) .* (normpdf((Y-yhat)/sigma)/sigma)));
    ll = - ll;   
end

opts = optimoptions('fminunc','MaxFunctionEvaluations',1e6...
    , 'MaxIterations',1e6);
[xMLE,~,~,~,~,H] = fminunc(@calclike, sv,opts);
xSE = sqrt(diag(H \ eye(size(H))));
estMLE = vecparm(xMLE);
estSE = vecparm(xSE);
estMLE.SE = estSE.b;
estMLE.z = estMLE.b ./ estMLE.SE;
estMLE.p = (1 - normcdf(abs(estMLE.z))) * 2;
estMLE.bVCOV = H \ eye(size(H));

end
