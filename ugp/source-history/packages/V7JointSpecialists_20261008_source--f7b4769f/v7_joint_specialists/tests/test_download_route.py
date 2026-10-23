import importlib.util
import unittest
from fixtures import ROOT
import sys
sys.path.insert(0,str(ROOT/'v7_joint_specialists/ops'))

class RouteChecks(unittest.TestCase):
    def test_training_launch_requires_matching_successful_real_verification(self):
        self.assertIsNotNone(importlib.util.find_spec('await_verified'),'Verified launch controller missing')
        from await_verified import verified_status
        self.assertFalse(verified_status({'pid':100,'phase':'command_audit'},100))
        self.assertTrue(verified_status({'pid':100,'phase':'verification_complete'},100))
        with self.assertRaises(ValueError):verified_status({'pid':99,'phase':'verification_complete'},100)
        with self.assertRaises(RuntimeError):verified_status({'pid':100,'phase':'failed','error':'alignment'},100)

    def test_only_public_dataset_tls_hosts_allowed(self):
        self.assertIsNotNone(importlib.util.find_spec('download_route'),'Private secure recording route missing')
        from download_route import allowed_target
        self.assertEqual(allowed_target('huggingface.co:443'),('huggingface.co',443))
        self.assertEqual(allowed_target('us.aws.cdn.hf.co:443'),('us.aws.cdn.hf.co',443))
        for target in ('huggingface.co:80','localhost:443','huggingface.co.attacker.test:443','127.0.0.1:443','example.com:443'):
            with self.assertRaises(ValueError):allowed_target(target)

if __name__=='__main__':unittest.main()
