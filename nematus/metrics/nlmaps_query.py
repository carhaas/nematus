
import subprocess
from threading import Timer
from ..util import *
import logging
from pymemcache.client import base
import mmh3

def get_nlmaps_answer(sample_text, model_options, client, ramp_answer_dict, source_text=""):
    use_memcache = model_options["use_memcache"]
    saveto = model_options["saveto"]
    int_1, int_2 = mmh3.hash64(sample_text)
    sample_text_hashed = str(int_1) + str(int_2)
    sample_answer = None
    if use_memcache:
        try:
            sample_answer = client.get('a:%s' % sample_text_hashed)
            logging.info('memcached answer retrieved: %s' % sample_answer)
        except:
            logging.info("memcache operation failed: retrieve answer")
    if sample_answer is None: #then we either aren't using memcache or this answer has not been cached yet
        if 'a:%s' % sample_text in ramp_answer_dict:
            sample_answer = ramp_answer_dict['a:%s' % sample_text]
            logging.info('ramp_answer_dict answer retrieved')
        else:
            # print sample to file
            write_list_to_file([sample_text], saveto + '.ramp.lin')

            # call execution script
            ramp_process = subprocess.Popen([model_options['ramp_execution_script'], saveto],
                                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            kill = lambda process: process.kill()
            ramp_timer = Timer(300, kill, [ramp_process])
            try:
                ramp_timer.start()
                stdout, stderr = ramp_process.communicate()
            except:
                logging.info(
                    "Timeout for the following translation: %s ||| %s" % (source_text, sample_text))
                return "timeout"
            finally:
                ramp_timer.cancel()

            # read in answers
            try:
                sample_answer = read_lines_in_list(saveto + '.ramp.answer')[0]
                if use_memcache:
                    try:
                        client.set('a:%s' % sample_text_hashed, sample_answer)
                        logging.info('memcached answer set')
                    except:
                        logging.info("memcache operation failed: set answer")
                else:
                    ramp_answer_dict['a:%s' % sample_text] = sample_answer
            except IndexError:
                pass
    return sample_answer