function estMLE = OrderedProbit(Y,X,controlvariables)
Yind = bsxfun(@eq, Y, unique(Y)');
[n, J] = size(Yind);
nx = size(X,2);

cut_sv = norminv(cumsum(mean(Yind(:,1:end-1))));
cut_sv(2:end) = log(cut_sv(2:end) - cut_sv(1:(end-1)));
%cut_sv = norminv(mean(Yind));
%cut_sv = cut_sv(1:end-1);
%cut_sv(2:end) = cut_sv(2:end) - cut_sv(1:(end-1));
%cut_sv(2:end) = log(cut_sv(2:end));

sv = [zeros([nx 1]); cut_sv']; 

function est = vecparm(x)
    est = struct();
    est.b = x(1:nx);
    cutpoints = x((nx+1):end);
    cutpoints(2:end) = exp(cutpoints(2:end));
    cutpoints = cumsum(cutpoints);
    est.a = [-inf; cutpoints; inf];
end

function [ll,pr] = calclike(x)
    est = vecparm(x);
    yhat = X*est.b;
    pr_base = normcdf(bsxfun(@minus,est.a',yhat));
    pr = pr_base(:,2:end) - pr_base(:,1:end-1);
    ll = sum(log(pr(Yind==1)));
    ll = -ll;
end

opts = optimoptions('fminunc','MaxFunctionEvaluations',1e6...
    , 'MaxIterations',1e6);
[xMLE,~,~,~,~,H] = fminunc(@calclike, sv,opts);
xSE = sqrt(diag(H \ eye(size(H))));
estMLE = vecparm(xMLE);

function pr = calcpr(X,est)
     yhat = X*est.b;
     pr_base = normcdf(bsxfun(@minus,est.a',yhat));
     pr = pr_base(:,2:end) - pr_base(:,1:end-1);
end

AME = zeros([nx J]);
pr0 = calcpr(X,estMLE);
for k=1:nx
    if isequal(unique(X(:,k)),[0;1])
        Xold = X;
        Xold(:,k) = 0;
        Xnew = X;
        Xnew(:,k) = 1;
        AME(k,:) = mean(calcpr(Xnew,estMLE) - calcpr(Xold,estMLE));
    else
        Xnew = X;
        if numel(unique(X(:,k))) < 50
            dx =1;
        else
            dx = .00001;
        end
        Xnew(:,k) = Xnew(:,k) + dx;
        AME(k,:) = mean( (calcpr(Xnew,estMLE) - pr0)./dx);
    end
end
    

estSE = vecparm(xSE);
estMLE.b = estMLE.b;

estMLE.SE = estSE.b;

estMLE.AME = array2table(AME, 'VariableNames', strcat('AME',string(0:(J-1)))...
    ,'RowNames',controlvariables);

end