import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil, log, sqrt, copysign, floor
from statsmodels.stats.proportion import proportion_confint

PREC = .00000001
class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, nA, pABar
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, nA, pABar

    def certify_dipole(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        assert n %2 == 0
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
 
        estS, estN = self._sample_noise_dipole(x, n, batch_size)
        nS = estS[cAHat].item()
        nN = estN[cAHat].item()
        C_S_bar = self._lower_confidence_bound(nS, int(n/2), alpha/2)
        C_N_bar = self._lower_confidence_bound(nN, int(n/2), alpha/2)
        radius = self._dipole_certified_radius(C_S_bar, C_N_bar)
        if  (radius > 0.0):
            return cAHat, radius, nS, nN, C_S_bar, C_N_bar
        else:
            return Smooth.ABSTAIN, 0.0, nS, nN, C_S_bar, C_N_bar

    def certify_second_order(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int):
        self.base_classifier.eval()
        counts_selection = self._sample_noise(x, n0, batch_size)
        cAHat = counts_selection.argmax().item()
        counts_estimation, grad_squared = self._sample_noise_with_grad_squared(x, n, cAHat, batch_size)
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha/2)
        gradSquaredBar, meaningful = self._grad_squared_upper_bound(grad_squared, n, x.nelement(), alpha/2, pABar)

        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0, nA, pABar, copysign(sqrt(abs(grad_squared)),grad_squared) , sqrt(gradSquaredBar), meaningful
        else:
            radius  = self._solve_for_radius_second_order(pABar, sqrt(gradSquaredBar))
            return cAHat, radius, nA, pABar, copysign(sqrt(abs(grad_squared)),grad_squared), sqrt(gradSquaredBar), meaningful

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts
    def _sample_noise_dipole(self, x: torch.tensor, num: int, batch_size, include_totals=False):
        """ Sample the base classifier's prediction under noisy corruptions of the input x, where the noise is symmetric

        :param x: the input [channel x width x height]
        :param num: number of samples to collect. 
        :param batch_size: 
        :return: a tuple, first element is an ndarray[int] of length num_classes containing
                 the per-class counts where both sides of the dipole belong to the class,
                 second is where only one element belongs, third is total count estimates (if flagged)
        """
        with torch.no_grad():
            counts_S = np.zeros(self.num_classes, dtype=int)
            counts_N = np.zeros(self.num_classes, dtype=int)
            count_total = np.zeros(self.num_classes, dtype=int)
            assert num %2 == 0, "Need total samples to be divisible by two"

            for _ in range(ceil(num / (2*batch_size))):
                this_batch_size = min(batch_size, int(num/2))
                num -= this_batch_size*2

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions_l = self.base_classifier(batch + noise).argmax(1)
                predictions_r = self.base_classifier(batch - noise).argmax(1)
                new_S, new_N = self._count_arrs_dipole(predictions_l.cpu().numpy(), predictions_r.cpu().numpy(), self.num_classes)
                counts_S += new_S
                counts_N += new_N
                if (include_totals):
                    count_total += self._count_arr(predictions_l.cpu().numpy(), self.num_classes) + self._count_arr(predictions_r.cpu().numpy(), self.num_classes)
            if (include_totals):
                return counts_S, counts_N, count_total 
            else:
                return counts_S, counts_N 

    def _sample_noise_with_grad_squared(self, x: torch.tensor, num: int,  correct_class: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        assert num %2 == 0, "Need total samples to be divisible by two"
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            all_normsquared_ests = []
            for _ in range(ceil(num / (2*batch_size))):
                this_batch_size = min(batch_size, int(num/2))
                num -= this_batch_size*2

                batchA = x.repeat((this_batch_size, 1, 1, 1))
                noiseA = torch.randn_like(batchA, device='cuda') * self.sigma
                predictionsA = self.base_classifier(batchA + noiseA).argmax(1)
                countsA = self._count_arr(predictionsA.cpu().numpy(), self.num_classes)
                counts += countsA
                batchB = x.repeat((this_batch_size, 1, 1, 1))
                noiseB = torch.randn_like(batchB, device='cuda') * self.sigma
                predictionsB = self.base_classifier(batchB + noiseB).argmax(1)
                countsB = self._count_arr(predictionsB.cpu().numpy(), self.num_classes)
                counts += countsB
                predictionsTrue = (predictionsA == correct_class) & (predictionsB == correct_class)
                grad_vecs = torch.zeros(noiseA.shape,  device='cuda')
                grad_vecs[predictionsTrue] = noiseA[predictionsTrue]  * noiseB[predictionsTrue]
                normsquared_ests  =  grad_vecs.sum(dim=3).sum(dim=2).sum(dim=1)
                all_normsquared_ests.append(normsquared_ests)
            grad_squared = torch.cat(all_normsquared_ests).mean()/(self.sigma**4)
            return counts, grad_squared.cpu().numpy()

    def _count_arrs_dipole(self, arr_l: np.ndarray, arr_r: np.ndarray, length: int) -> np.ndarray:
        counts_S = np.zeros(length, dtype=int)
        counts_N = np.zeros(length, dtype=int)
        c = arr_l.size
        for idxidx in range(c):
            idx_l = arr_l[idxidx]
            idx_r = arr_r[idxidx]
            if (idx_l == idx_r):
                counts_S[idx_l] += 1
            else:
                counts_N[idx_l] += 1
        return counts_S, counts_N 

    def _dipole_certified_radius(self, C_S_bar, C_N_bar):
        if (self._dipole_certified_score( C_S_bar, C_N_bar,0.0) < 0.5):
            return 0.0
        else:
            R_lower = 0.0
            R_upper = self.sigma
            while  (self._dipole_certified_score( C_S_bar, C_N_bar,R_upper) > 0.5):
                R_upper = R_upper*2
            while (R_upper-R_lower > PREC*self.sigma):
                R_mid  = (R_upper+R_lower)/2
                if (self._dipole_certified_score( C_S_bar, C_N_bar,R_mid) > 0.5):
                    R_lower=R_mid
                else:
                    R_upper=R_mid
            return R_lower
    def _dipole_certified_score(self, C_S_bar, C_N_bar, R):
        Rs = R/self.sigma
        return norm.cdf(norm.ppf(C_N_bar)-Rs)+ norm.cdf(norm.ppf((C_S_bar+1.)/2.)-Rs)- norm.cdf(norm.ppf((-C_S_bar+1.)/2.)-Rs)

    def _grad_squared_upper_bound(self,grad_squared, num_samples, dim, conf, score_lb):
        assert num_samples %2 == 0, "Need total samples to be divisible by two"
        n = num_samples / 2
        if (dim*n >= -2 * log(conf)):
            t = 4* self.sigma**2 * sqrt(-1* dim* log(conf)/n)
        else:
            t = -4*sqrt(2)* self.sigma**2 * log(conf) / n
        ub = grad_squared + t/ (self.sigma**4)
        meaningful = 1
        if ub < 0:
            ub = 0
        elif ub > self.max_possible_grad(score_lb)**2:
            ub = self.max_possible_grad(score_lb)**2
            meaningful = 0
        return ub, meaningful

    def _a_prime_lhs(self,a,score_lb):
        return norm.pdf(norm.ppf(a)) - norm.pdf(norm.ppf(a+score_lb))
    def _solve_for_a_prime(self,score_lb, grad_norm_ub):
        a_lb = 0.
        a_ub = 1. - score_lb
        while (a_ub-a_lb > PREC):
            a_mid = (a_lb + a_ub)/2
            if (self._a_prime_lhs(a_mid,score_lb) > -self.sigma * grad_norm_ub):
                a_ub = a_mid
            else:
                a_lb = a_mid
        return a_lb

    def _solve_for_radius_second_order(self,score_lb, grad_norm_ub):
        if (self._certified_score_second_order(score_lb, grad_norm_ub,0.0) < 0.5):
            return 0.0
        else:
            R_lower = self.sigma*norm.ppf(score_lb)
            R_upper = self.sigma*norm.ppf(score_lb+ .5*(1-score_lb))
            while (R_upper-R_lower > PREC*self.sigma):
                R_mid  = (R_upper+R_lower)/2
                if (self._certified_score_second_order(score_lb, grad_norm_ub,  R_mid) > 0.5):
                    R_lower=R_mid
                else:
                    R_upper=R_mid
            return R_lower

    def _certified_score_second_order(self,score_lb, grad_norm_ub, radius):
        a_prime = self._solve_for_a_prime(score_lb, grad_norm_ub)
        return norm.cdf(norm.ppf(score_lb+a_prime)-radius/self.sigma)- norm.cdf(norm.ppf(a_prime)-radius/self.sigma)

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts
    def max_possible_grad(self,score_lb):
        return  norm.pdf(norm.ppf(score_lb))/self.sigma
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
