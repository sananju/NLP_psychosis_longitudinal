%% PLS script to relate NLP measures to symptom and functioning scores
% Dr Sarah Morgan, 22/07/2025
% Adapted from code by Prof. Petra Vertes
% (see https://doi.org/10.1098/rstb.2015.0362)

X=nlp_93; % Predictors- 8 NLP metrics for 93 subjects
Y=panss; % Response variable- 7 PANSS and functioning scores for 93 subjects

% z-score:
X=zscore(X);
Y=zscore(Y);

%perform full PLS and plot variance in Y explained by top 8 components
%typically top 2 or 3 components will explain a large part of the variance
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Y);
dim=8;
plot(1:dim,cumsum(100*PCTVAR(2,1:dim)),'-o','LineWidth',1.5,'Color',[140/255,0,0]);
set(gca,'Fontsize',14)
xlabel('Number of PLS components','FontSize',14);
ylabel('Percent Variance Explained in Y','FontSize',14);
grid on

dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Y,dim); % no need to do this but it keeps outputs tidy

% permutation testing to assess significance of PLS result as a function of
% the number of components (dim) included:

clear R p Rsquared Rsq
rep=10000;
for dim=1:8
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Y,dim);
temp=cumsum(100*PCTVAR(2,1:dim));
Rsquared = temp(dim);
    for j=1:rep
        %j
        order=randperm(size(Y,1));
        Yp=Y(order,:);

        [XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Yp,dim);

        temp=cumsum(100*PCTVAR(2,1:dim));
        Rsq(j) = temp(dim);
    end
dim
R(dim)=Rsquared
p(dim)=length(find(Rsq>=Rsquared))/rep
end
figure
plot(1:dim, p,'ok','MarkerSize',8,'MarkerFaceColor','r');
xlabel('Number of PLS components','FontSize',14);
ylabel('p-value','FontSize',14);
grid on

dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Y,dim);