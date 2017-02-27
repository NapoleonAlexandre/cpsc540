%% garbage

%for j=1:size(ind,1)
    %    SIGMA{i} = SIGMA{i} + (1/(n*theta(i)))*((Xtrain(ind(j),:)'-mu{i})*(Xtrain(ind(j),:)'-mu{i})');
    %end
    
    
    %Compute cavariance matrix SIGMA_i
        model.SIGMA{i} = zeros(d);
        SIGMA{i}
        for j=1:size(ind,1)
            model.SIGMA{i} = model.SIGMA{i} + (1/(n*model.theta(i)+sum(r{i})))...
                *((Xtrain(ind(j),:)'-model.mu{i})*(Xtrain(ind(j),:)'-mu{i})');...
        end
        for j=1:t
            model.SIGMA{i} = model.SIGMA{i} + (1/(n*model.theta(i)+sum(r{i})))...
            *spdiags(r{i},0,t,t)*((Xunlabeled'-model.mu{i})*(Xtrain(ind(j),:)'-mu{i})'
            );
        end
        
        %Compute Q
        Q1= sum(log(mvnpdf(Xtrain(ind,:),model.mu{i}',model.SIGMA{i})));
        Q2 = sum(r(:,i).*log(mvnpdf(Xunlabeled,model.mu{i}',model.SIGMA{i})));
        i
        Q1
        Q2
    end
    Q
    dQ = abs(Qold-Q)/abs(Q);
    dQ