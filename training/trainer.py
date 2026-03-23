"""
training/trainer.py  --  PPO Trainer

VÁLTOZÁSOK:
  [RF-2 FIX] StepLR(step_size=1_000_000) → CosineAnnealingLR(T_max=500)
    Az eredeti step_size hatékonyan végtelen volt (~2 milliárd epizód / decay),
    mert scheduler.step() minden update()-ben hívódott, nem epizódonként.
    CosineAnnealingLR simán csökkenti az LR-t 500 update-ciklus alatt
    (≈ 1M epizód 2048 buffer_size mellett), ami PPO-hoz ideális.
"""
import collections, numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from .buffer import PPOBuffer

class PPOTrainer:
    CLIP_EPS=0.2; PPO_EPOCHS=8; MINIBATCH=256; VALUE_COEF=0.5
    ENTROPY_COEF=0.01; ENTROPY_FINAL=0.001; ENTROPY_DECAY=900_000  # [FIX-3: ~30M ep / 2048 buffer × 60 minibatch]
    MAX_GRAD_NORM=0.5; GAMMA=0.99; GAE_LAM=0.95
    # LR scheduler paraméterek – T_max=500 update ≈ 1M epizód (2048 buffer_size mellett)
    LR_T_MAX=14_648; LR_ETA_MIN_RATIO=0.05  # min LR = alap LR * 0.05  [FIX-1: 30M ep / 2048 buffer]

    def __init__(self, model, lr=3e-4, device=None):
        self.model=model; self.device=device or torch.device('cpu')
        self._lr=lr  # eltároljuk eta_min kiszámításához
        self.use_amp=(self.device.type=='cuda')
        self.optimizer=optim.Adam(model.parameters(),lr=lr,eps=1e-5)
        # [RF-2 FIX] CosineAnnealingLR: lr lineárisan csökken T_max update alatt,
        # majd felmegy, majd újra csökken – stabilabb tanulási dinamikát ad PPO-hoz.
        self.scheduler=optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.LR_T_MAX,
            eta_min=lr * self.LR_ETA_MIN_RATIO,
        )
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.use_amp)
        self._total_updates=0

    def _entropy_coef(self):
        progress=min(self._total_updates/self.ENTROPY_DECAY,1.0)
        return self.ENTROPY_COEF+(self.ENTROPY_FINAL-self.ENTROPY_COEF)*progress

    def update(self, buffer, last_value: float = 0.0):
        """
        PPO update.

        Paraméterek:
            buffer:     PPOBuffer – a gyűjtött tapasztalatok
            last_value: [RF-3 FIX] V(s_{T+1}) bootstrap érték a buffer
                        utolsó lépése UTÁNI állapothoz.
                        - Terminális lépés esetén: 0.0 (default, helyes)
                        - Nem-terminális esetén: a learner V(s) értéke
                          az utolsó env állapothoz (runner.py adja át)
        """
        if len(buffer)<4: return {}
        advantages,returns=buffer.compute_gae(self.GAMMA,self.GAE_LAM,
                                               last_value=last_value)
        dev=self.device
        old_log_probs=torch.stack(buffer.log_probs).to(dev)
        actions_t=torch.stack(buffer.actions).view(-1).to(dev)
        states_t=torch.stack(buffer.states).to(dev)
        advantages=advantages.to(dev); returns_t=returns.to(dev)
        metrics=collections.defaultdict(list)
        for _ in range(self.PPO_EPOCHS):
            idx=torch.randperm(len(buffer))
            for start in range(0,len(buffer),self.MINIBATCH):
                mb=idx[start:start+self.MINIBATCH]
                if len(mb)<2: continue
                mb_states=states_t[mb]; mb_legal=[buffer.legal_actions[i.item()] for i in mb]
                mb_actions=actions_t[mb]; mb_old_lp=old_log_probs[mb]
                mb_adv=advantages[mb]; mb_ret=returns_t[mb]
                with torch.amp.autocast('cuda',enabled=self.use_amp):
                    new_lp,entropy,new_val=self.model.evaluate_actions(mb_states,mb_legal,mb_actions)
                    ratio=torch.exp(new_lp-mb_old_lp)
                    surr1=ratio*mb_adv
                    surr2=torch.clamp(ratio,1.0-self.CLIP_EPS,1.0+self.CLIP_EPS)*mb_adv
                    actor_loss=-torch.min(surr1,surr2).mean()
                    critic_loss=F.huber_loss(new_val,mb_ret)
                    ent_loss=-entropy.mean()
                    loss=actor_loss+self.VALUE_COEF*critic_loss+self._entropy_coef()*ent_loss
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.MAX_GRAD_NORM)
                self.scaler.step(self.optimizer); self.scaler.update()
                self._total_updates+=1
                metrics['actor'].append(actor_loss.item())
                metrics['critic'].append(critic_loss.item())
                metrics['entropy'].append(-ent_loss.item())
        self.scheduler.step(); buffer.reset()
        return {k: float(np.mean(v)) for k,v in metrics.items()}

    def state_dict(self):
        return {'optimizer':self.optimizer.state_dict(),'scheduler':self.scheduler.state_dict(),
                'scaler':self.scaler.state_dict(),'total_updates':self._total_updates}

    def load_state_dict(self, d):
        if 'optimizer' in d:
            try:
                self.optimizer.load_state_dict(d['optimizer'])
            except (KeyError, ValueError) as e:
                # Architektúra változáskor (pl. v4.1→v4.2 GRU rétegek hozzáadva)
                # az optimizer parameter group mérete nem egyezik → kihagyjuk.
                # Az optimizer friss állapotból indul, a model weights megmaradnak.
                import logging
                logging.getLogger('PokerAI').warning(
                    f'Optimizer state nem töltve be (architektúra mismatch): {e}'
                )
        if 'scheduler' in d:
            try:
                self.scheduler.load_state_dict(d['scheduler'])
            except (KeyError, ValueError):
                # Régi StepLR checkpoint → CosineAnnealingLR nem kompatibilis;
                # egyszerűen kihagyjuk (scheduler nulláról indul).
                pass
        if 'scaler'    in d: self.scaler.load_state_dict(d['scaler'])
        self._total_updates=d.get('total_updates',0)
