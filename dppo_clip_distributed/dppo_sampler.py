import multiprocessing as mp


def write_log(text: str):
    pass
    # with open('sampler_log.txt', 'a') as f:
    #     f.write(text+'\n')


class DPPOSampler:
    def __init__(self, create_env_func):
        self.env_builder = create_env_func
        self.env = None

    def init_components(self):
        self.env = self.env_builder()

    def run(self, pipe, should_stop: mp.Event):
        self.init_components()
        write_log('---------------' * 10)
        state = self.env.reset()
        done, reward, _ = True, 0, {}
        write_log('going into while')
        while not should_stop.is_set():
            write_log('sending data')
            pipe.send((state, reward, done, _))
            write_log('recving data')
            action = pipe.recv()
            state, reward, done, _ = self.env.step(action)
            if done:
                state = self.env.reset()


if __name__ == '__main__':
    from rlzoo.common.env_wrappers import build_env
    import multiprocessing as mp


    def build_func():
        return build_env('CartPole-v0', 'classic_control')


    remote_a, remote_b = mp.Pipe()
    should_stop = mp.Event()
    should_stop.clear()

    spl = DPPOSampler(build_func)
    p = mp.Process(target=spl.run, args=(remote_a, should_stop))
    p.daemon = True
    p.start()

    while True:
        s, r, d, _ = remote_b.recv()
        remote_b.send(1)
        print(s, r, d, _)
