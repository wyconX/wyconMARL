class wyPolicy():

    def __init__(self, ep_min=0.01, ep_decay=0.0001, esp_total=1000):
        self.ep_min = ep_min
        self.ep_decay = ep_decay
        self.eps_total = esp_total

    def epsilon(self, step, ep_max=1):
        if self.eps_total == 1:
            return self.ep_min
        eps = max(self.ep_min, ep_max - (ep_max - self.ep_min)*step/self.eps_total )
        return eps