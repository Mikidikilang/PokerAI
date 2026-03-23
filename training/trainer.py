"""
training/trainer.py  --  PPO Trainer

VÁLTOZÁSOK:
  [FIX-1] CosineAnnealingLR T_max=500 → T_max=14_648
    Az eredeti T_max=500 ~1M epizódonként visszaugrott a max LR-re (3e-4).
    30M epizód alatt ~29 teljes ciklust jelentett → catastrophic forgetting
    ismétlődött minden ciklusnál.
    Javítás: T_max = 30_000_000 // 2048 buffer_size = 14_648 update.
    Az LR most egyirányban csökken a teljes tréning időtartama alatt.

  [FIX-3] ENTROPY_DECAY egységhiba javítva: 30_000_000 → 900_000
    _total_updates MINIBATCH-ITERÁCIÓKAT számol, nem epizódokat.
    30M epizód végén total_updates ~937_000 → progress volt ~3.1%.
    Javítás: ENTROPY_DECAY = 900_000 (30M ep / 2048 buffer * 60 minibatch).
    Az entrópia-együttható most a tréning végén ténylegesen ENTROPY_FINAL-ra csökken.

  [RF-2 FIX] StepLR → CosineAnnealingLR (megtartva, T_max javítva fentebb)
"""
import collections, numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from .buffer import PPOBuffer

class PPOTrainer:
    CLIP_EPS=0.2; PPO_EPOCHS=8; MINIBATCH=256; VALUE_COEF=0.5
    ENTROPY_COEF=0.01; ENTROPY_FINAL=0.001

    # [FIX-3] Egységhiba javítva: 30_000_000 (epizód) → 900_000 (minibatch iter)
    # Magyarázat: 30M ep / 2048 buffer * ~60 minibatch/update ≈ 878k → kerek 900k
    # Így progress=100% ténylegesen ~30M ep körül következik be.
    ENTROPY_DECAY = 900_000

    MAX_GRAD_NORM=0.5; GAMMA=0.99; GAE_LAM=0.95

    # [FIX-1] T_max javítva: 500 → 14_648
    # Számítás: 30_000_000 ep / 2048 buffer_size = 14_648 update
    # Így az LR egyirányban, monoton csökken a teljes tréning alatt –
    # nincs több catastrophic forgetting a ciklusok miatt.
    LR_T_MAX = 14_648
    LR_ETA_MIN_RATIO = 0.05  # min LR = alap LR * 0.05

    def __init__(self, model, lr=3e-4, device=None):
        self.model=model; self.device=device or torch.device('cpu')
        self._lr=lr
        self.use_amp=(self.device.type=='cuda')
        self.optimizer=optim.Adam(model.parameters(),lr=lr,eps=1e-5)

        # [FIX-1] CosineAnnealingLR helyes T_max-szal:
        # Az LR 3e-4-ről smoothan csökken lr*0.05-re 14_648 update alatt,
        # majd ott marad – nem ugrik vissza a maximumra.
        self.scheduler=optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.LR_T_MAX,
            eta_min=lr * self.LR_ETA_MIN_RATIO,
        )
        self.scaler=torch.amp.GradScaler('cuda',enabled=self.use_amp)
        self._total_updates=0

    def _entropy_coef(self):
        # [FIX-3] ENTROPY_DECAY most minibatch-iteráció egységben van,
        # ugyanolyan egységben mint _total_updates → progress helyes.
        progress=min(self._total_updates/self.ENTROPY_DECAY,1.0)
        return self.ENTROPY_COEF+(self.ENTROPY_FINAL-self.ENTROPY_COEF)*progress

    def update(self, buffer, last_value: float = 0.0):
        """
        PPO update.

        Paraméterek:
            buffer:     PPOBuffer – a gyűjtött tapasztalatok
            last_value: V(s_{T+1}) bootstrap érték a buffer utolsó lépése
                        UTÁNI állapothoz. Terminális esetén 0.0 (default).
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
                import logging
                logging.getLogger('PokerAI').warning(
                    f'Optimizer state nem töltve be (architektúra mismatch): {e}'
                )
        if 'scheduler' in d:
            try:
                self.scheduler.load_state_dict(d['scheduler'])
            except (KeyError, ValueError):
                # Régi StepLR vagy T_max=500-as checkpoint → kihagyjuk,
                # a scheduler T_max=14_648-cal indul nulláról. Helyes viselkedés.
                pass
        if 'scaler' in d: self.scaler.load_state_dict(d['scaler'])
        self._total_updates=d.get('total_updates',0)
