import importlib.util
import tempfile
import unittest
from pathlib import Path
import zipfile
from fixtures import ROOT

class RecordingChecks(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('recording_source'),'Missing-recording recovery not implemented')
        from recording_source import recording
        return recording

    def test_reuses_original_and_only_extracts_requested_bag(self):
        recording=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);original=root/'original/ForestEnv/Data_anymal/P2000/rosbags/P2000.bag';original.parent.mkdir(parents=True);original.write_bytes(b'original')
            def forbidden(*args):raise AssertionError('Existing recordings must not download')
            with recording(root/'original','ForestEnv','P2000',root/'scratch',forbidden) as p:self.assertEqual(p.read_bytes(),b'original')
            def fetch(env,run,path):
                with zipfile.ZipFile(path,'w') as z:
                    z.writestr('rosbags/'+run+'.bag',b'raw commands');z.writestr('unused.txt',b'unused')
            with recording(root/'original','ForestEnv','P2003',root/'scratch',fetch) as p:
                self.assertEqual(p.read_bytes(),b'raw commands');downloaded=p
            self.assertFalse(downloaded.exists());self.assertTrue(original.exists())

    def test_wrong_recording_or_unsafe_identity_rejected(self):
        recording=self.api()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def fetch(env,run,path):
                with zipfile.ZipFile(path,'w') as z:z.writestr('wrong.bag',b'bad')
            with self.assertRaises(ValueError):
                with recording(root/'original','ForestEnv','P2003',root/'scratch',fetch):pass
            with self.assertRaises(ValueError):
                with recording(root,'../other','P2003',root,fetch):pass

if __name__=='__main__':unittest.main()
