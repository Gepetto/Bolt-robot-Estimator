import numpy as np

class DataOut():
    def __init__(self, k:str, directory='./Données cancer niels'):
        self.k = k
        self.iter_nosave = 0
        self.iter = 0
        self.dir = directory + '/' + k + '/'
        
        # data
        self.t_vec = []
        self.all_pos_l = []
        self.all_theta_l = []
        self.all_vel_l = []
        self.all_q_l = []
        self.all_qd_l = []
        self.all_w_l = []
        self.all_tau_l = []
        self.all_lcf = []
        self.all_rcf = []
        self.all_left_contact = []
    
    def store(self, t, pos, theta, vel, q, qd, w, tau, lcf, rcf, left_contact):
        self.t_vec.append(t)
        self.all_pos_l.append(pos.copy())
        self.all_theta_l.append(pos.copy())
        self.all_vel_l.append(vel.copy())
        self.all_q_l.append(q.copy())
        self.all_qd_l.append(qd.copy())
        self.all_w_l.append(w.copy())
        self.all_tau_l.append(tau.copy())
        self.all_lcf.append(lcf.copy())
        self.all_rcf.append(rcf.copy())
        self.all_left_contact.append(left_contact)
        
        self.iter_nosave += 1
        self.iter +=1
        
    def autosave(self, n):
        # save data every n
        if self.iter_nosave > n :
            self.save()
            self.iter_nosave=0
            return True
        return False
        
    
    def save(self):
        np.save(self.dir + 'T_array_'+self.k, np.array(self.t_vec))
        np.save(self.dir + 'X_array_'+self.k, np.array(self.all_pos_l))
        np.save(self.dir + 'Theta_array_'+self.k, np.array(self.all_theta_l))
        np.save(self.dir + 'V_array_'+self.k, np.array(self.all_vel_l))
        np.save(self.dir + 'Q_array_'+self.k, np.array(self.all_q_l))
        np.save(self.dir + 'Qd_array_'+self.k, np.array(self.all_qd_l))
        np.save(self.dir + 'W_array_'+self.k, np.array(self.all_w_l))
        np.save(self.dir + 'Tau_array_'+self.k, np.array(self.all_tau_l))
        np.save(self.dir + 'LCF_array_'+self.k, np.array(self.all_lcf))
        np.save(self.dir + 'RCF_array_'+self.k, np.array(self.all_rcf))
        np.save(self.dir + 'C_array_'+self.k, np.array(self.all_left_contact))