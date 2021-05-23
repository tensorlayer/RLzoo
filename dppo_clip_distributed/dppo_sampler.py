import multiprocessing as mp


class DPPOSampler:
    def __init__(self, create_env_func):
        self.env = create_env_func()

    def run(self, pipe, should_stop: mp.Event):
        state = self.env.reset()
        done, reward, _ = True, 0, {}
        while not should_stop.is_set():
            pipe.send((state, reward, done, _))
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
