#Inference for binomial proportion

load algae.txt;

%INITIALIZATION OF VARIABLES
X = sum(algae(:) == 1);
Success = sum(algae(:) == 1 ) / ((sum(algae(:) == 1)+(sum(algae(:) == 0))));
A = 2;
B = 10;
N = length(algae);

%DISCRETE VALUE CALCULATION
Prior_Value = ((Success^(A-1)) * ((1-Success)^(B-1) )) / (beta(A,B));
Likelihood_Value = ( (prod(N-X+1:N)) * (Success^X) * ((1-Success)^(N-X)) ) / ( prod(1:X) ) ;
Posterior_Value = (Success^(X+A-1))*((1-Success)^(N+B-X-1))/beta(X+A,N+B-X);

%PLOTTING
Y = linspace(0,1);
prior = betapdf(Y,A,B);
plot(Y,prior)
hold on
posterior = betapdf(Y,X+A,N+B-X);
plot(Y,posterior)
xlabel('Y')
ylabel('Probability Density')
legend('prior','posterior')
hold off
%POINT ESTIMATES (Mean/Mode/Standard Deviation - Ref: goo.gl/mgauwh)
mean_posterior_distribution = (X+A)/(A+B+N);
mode_posterior_distribution = (X+A-1)/(A+B+N-2);
std_dev_posterior_distribution = (((X+A)*(N+B-X))/(((A+B+N)^2)*(A+B+N+1)))^0.5;

%95 Percent Credible Interval
alpha = 0.05;
lower_posterior = betainv(alpha/2, X+A, N+B-X);
upper_posterior = betainv(1-alpha/2, X+A, N+B-X);
lower_prior = betainv(alpha/2, A, B);
upper_prior = betainv(1-alpha/2, A, B);

%ANALYSIS OF SENSITIVITY
p_prior_mean = zeros(6,1);
p_posterior_mean = zeros(6,1);
p_posterior_mode = zeros(6,1);
p_posterior_std_dev = zeros(6,1);

a = zeros(6,1);
b = zeros(6,1);
p_lower_posterior = zeros(6,1);
p_upper_posterior = zeros(6,1);
a(1) = 1;
b(1) = 5;
for i=2 : 6
    a(i) = a(i-1)*2;
    b(i) = b(i-1)*2;
    p_prior_mean(i) = a(i)/(a(i)+b(i));
    p_posterior_mean(i) = (X+a(i))/(a(i)+b(i)+N);
    p_posterior_mode(i) = (X+a(i)-1)/(a(i)+b(i)+N-2);
    p_posterior_std_dev(i) = (((X+a(i))*(N+b(i)-X))/(((a(i)+b(i)+N)^2)*(a(i)+b(i)+N+1)))^0.5;
    p_lower_posterior(i) = betainv(alpha/2, X+a(i), N+b(i)-X);
    p_upper_posterior(i) = betainv(1-alpha/2, X+a(i), N+b(i)-X);
end
