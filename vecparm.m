function est = vecparm(THETA)

est = struct();
est.b = reshape(THETA(1:end-6), [], 3);
chol_cov = [THETA(end-5) 0 0;
            THETA(end-4) THETA(end-2) 0;
            THETA(end-3) THETA(end-1) THETA(end)];

est.cov = chol_cov * chol_cov';