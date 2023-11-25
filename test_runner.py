import unittest
import argparse

parser = argparse.ArgumentParser(description='IonImage embedding testrunner')


parser.add_argument('test_suite', type=str, const='all', nargs='?', default='all',
                    help='Which tests to run: "all", "dataloader", "crl", "lightningCRL", (default: "all")',)

args = parser.parse_args()
scenario = args.test_suite

test_suite = unittest.TestSuite()
loader = unittest.TestLoader()

if scenario == 'all':
    print('Running all tests')
    from tests import test_dataloader, test_crl, test_lightningCRL
    test_suite.addTests(loader.loadTestsFromModule(test_dataloader))
    test_suite.addTests(loader.loadTestsFromModule(test_crl))
    test_suite.addTests(loader.loadTestsFromModule(test_lightningCRL))

elif scenario == 'dataloader':
    print('Running dataloader tests')
    from tests import test_dataloader
    test_suite.addTests(loader.loadTestsFromModule(test_dataloader))

elif scenario == 'crl':
    print('Running crl tests')
    from tests import test_crl
    test_suite.addTests(loader.loadTestsFromModule(test_crl))

elif scenario == 'lightningCRL':
    print('Running lightningCRL tests')
    from tests import test_lightningCRL
    test_suite.addTests(loader.loadTestsFromModule(test_lightningCRL))

else:
    raise ValueError(f'ValueError: Testsuite "{scenario}" unknown. Please open the script help (python test_runner.py -h)')


runner = unittest.TextTestRunner()
result = runner.run(test_suite)
exit(0 if result.wasSuccessful() else 1)
