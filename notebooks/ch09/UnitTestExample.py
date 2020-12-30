from .CleanCode import DistributionAnalysis
import numpy as np
import scipy.stats as stat


def test_generate_boundaries():
    expected_low_norm = -2.3263478740408408
    expected_high_norm = 2.3263478740408408
    boundary_arguments = {'location': 0, 'scale': 1, 'arguments': ()}
    test_object = DistributionAnalysis(np.arange(0, 100), 10)
    normal_distribution_low = test_object._generate_boundaries(stat.norm,
                                                               boundary_arguments,
                                                               0.01)
    normal_distribution_high = test_object._generate_boundaries(stat.norm,
                                                                boundary_arguments,
                                                                0.99)
    assert normal_distribution_low == expected_low_norm, \
        'Normal Dist low boundary: {} does not match expected: {}' \
            .format(normal_distribution_low, expected_low_norm)
    assert normal_distribution_high == expected_high_norm, \
        'Normal Dist high boundary: {} does not match expected: {}' \
            .format(normal_distribution_high, expected_high_norm)


if __name__ == '__main__':
    test_generate_boundaries()
    print('tests passed')
