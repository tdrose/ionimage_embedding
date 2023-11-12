import unittest
import argparse

parser = argparse.ArgumentParser(description='IonImage embedding testrunner')


parser.add_argument('test_suite', type=str, const='all', nargs='?', default='all',
                    help='Which tests to run: "all", "dataloader", "clr", "lightningCLR", (default: "all")',)

args = parser.parse_args()
scenario = args.test_suite

test_suite = unittest.TestSuite()
loader = unittest.TestLoader()

if scenario == 'all':
    print('Running all tests')
    from tests import test_dataloader, test_clr, test_lightningCLR
    test_suite.addTests(loader.loadTestsFromModule(test_dataloader))
    test_suite.addTests(loader.loadTestsFromModule(test_clr))
    test_suite.addTests(loader.loadTestsFromModule(test_lightningCLR))

elif scenario == 'dataloader':
    print('Running dataloader tests')
    from tests import test_dataloader
    test_suite.addTests(loader.loadTestsFromModule(test_dataloader))

elif scenario == 'clr':
    print('Running clr tests')
    from tests import test_clr
    test_suite.addTests(loader.loadTestsFromModule(test_clr))

elif scenario == 'lightningCLR':
    print('Running lightningCLR tests')
    from tests import test_lightningCLR
    test_suite.addTests(loader.loadTestsFromModule(test_lightningCLR))

else:
    raise ValueError(f'ValueError: Testsuite "{scenario}" unknown. Please open the script help (python test_runner.py -h)')


runner = unittest.TextTestRunner()
result = runner.run(test_suite)
exit(0 if result.wasSuccessful() else 1)
