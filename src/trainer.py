import numpy as np
import torch
import os, time

from tqdm import tqdm

from utils import AverageMeter, evaluate, EmptyWith, log_print

class Trainer(object):
    def __init__(self,
        train_loader=None, 
        test_loader=None,
        model=None,
        mpsm=None,
        rfam_low=None,
        rfam_mid=None,
        rfam_high=None,
        optimizer=None, 
        ce_loss_fn=None,
        similarity_loss_fn=None,
        exp=None,
        similarity_loss_rate=10,
        log_interval=50,
        best_recond={"acc":0,"auc":0,"epoch":-1},
        save_dir="ckpt/test",
        exp_name="test"):
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.model=model
        self.mpsm=mpsm
        self.rfam_low=rfam_low
        self.rfam_mid=rfam_mid
        self.rfam_high=rfam_high
        self.optimizer=optimizer
        self.ce_loss_fn=ce_loss_fn
        self.similarity_loss_fn=similarity_loss_fn
        self.similarity_loss_rate=similarity_loss_rate
        self.log_interval = log_interval
        self.best_record = best_recond
        self.save_dir = save_dir
        self.exp_name = exp_name
        self.exp = exp
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.rfam_low = self.rfam_low.to(self.device)
        self.rfam_mid = self.rfam_mid.to(self.device)
        self.rfam_high = self.rfam_high.to(self.device)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)

    def train_epoch(self,epoch):
        self.model.train()

        train_loss_ce = AverageMeter()
        train_loss_similarity = AverageMeter()
        train_loss = AverageMeter()
        train_acc = AverageMeter()
        train_auc = AverageMeter()
        feature_norm = AverageMeter()

        start_time = time.time()
        previous_steps = (epoch-1) * len(self.train_loader)
        for batch_idx,(rgb_date, freq_data ,label, similarity_map) in enumerate(self.train_loader):
            my_step = previous_steps + batch_idx
            rgb_date = rgb_date.to(self.device)
            freq_data = freq_data.to(self.device)
            label = label.to(self.device)
            N = label.size(0)
            # forward
            self.optimizer.zero_grad()
            U1_low = self.model.block_1(rgb_date)
            U2_low = self.model.block_1(freq_data)
            A1_low, A2_low = self.rfam_low(U1_low, U2_low)
            x1 = U1_low * A1_low
            x2 = U2_low * A2_low

            U1_mid = self.model.block_2(x1)
            U2_mid = self.model.block_2(x2)
            A1_mid, A2_mid = self.rfam_mid(U1_mid, U2_mid)
            x1 = U1_mid * A1_mid
            x2 = U2_mid * A2_mid

            U1_high = self.model.block_3(x1)
            U2_high = self.model.block_3(x2)
            A1_high, A2_high = self.rfam_high(U1_high, U2_high)
            x1 = U1_high * A1_high
            x2 = U2_high * A2_high

            outputs_list = [(U1_low, A1_low), (U2_low, A2_low), (U1_mid, A1_mid), (U2_mid, A2_mid), (U1_high, A1_high),
                            (U2_high, A2_high)]

            predicted_similarity = self.mpsm.similarity_map(outputs_list)

            sim_loss = self.similarity_loss_fn(similarity_map.to(torch.float32).to(self.device), predicted_similarity.to(torch.float32).to(self.device))
            predicted_similarity = predicted_similarity.to(self.device)
            predicted_label = self.model.block_4(predicted_similarity)
            label = label.to(torch.long)
            print(label.shape, label, predicted_label.shape, predicted_label)

            cross_loss = self.ce_loss_fn(predicted_label, label)
            print("cross loss: ", cross_loss.item())
            print("sim loss: ", sim_loss.item())
            loss = cross_loss + (10 * sim_loss)
            print("loss: ", loss.item())
            loss = loss.to(torch.float32)



            # backward
            loss.backward()
            self.optimizer.step()

            outputs = predicted_label.data.cpu().numpy()
            label = label.data.cpu().numpy()
            acc, auc, tdr = evaluate(outputs,label)
            train_loss_ce.update(cross_loss.item(), N)
            train_loss_similarity.update(sim_loss.item(), N)
            train_loss.update(loss.item(), N)
            train_acc.update(acc, N)
            train_auc.update(auc, N)
            train_metrics = {
                "train_cross_loss": cross_loss.item(),
                "train_sim_loss": sim_loss.item(),
                "train_loss": loss.item(),
                "train_acc": acc,
                "train_auc": auc,
                "train_tdr": tdr
            }
            self.exp.log_metrics(train_metrics, step=my_step)
            if (batch_idx+1) % self.log_interval == 0:
                msg = '[{}][train] [epoch {}], [iter {} / {}], [loss {:.8f}],[loss ce{:.8f}],[loss similarity {:.8f}], [acc {:.5f}], [auc {:.5f}], [time used {:.1f}], [time left {:.1f}], [feature norm {}]'.format(
                self.exp_name, epoch, batch_idx, len(self.train_loader), train_loss.avg,train_loss_ce.avg,train_loss_similarity.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, (time.time()-start_time)/60/(batch_idx+1)*(len(self.train_loader)-batch_idx-1), feature_norm.avg)
                log_print(msg)
                msg = '[{}][train] [epoch {}], [loss {:.8f}], [loss ce {:.8f}],[loss similarity {:.8f}], [acc {:.5f}], [auc {:.5f}], [time {:.0f}], [lr {:.5f}], [feature norm {}]'.format(
                self.exp_name, epoch, train_loss.avg, train_loss_ce.avg,train_loss_similarity.avg, train_acc.avg, train_auc.avg, (time.time()-start_time)/60, self.optimizer.param_groups[0]['lr'], feature_norm.avg)
                log_print(msg)
        log_print("--------------------------------------------------------------------")
        log_print("similarity loss: {}".format(sim_loss.item()))
        log_print("cross entropy loss: {}".format(cross_loss.item()))
        log_print("loss: {}".format(loss.item()))

    def test_epoch(self,epoch):
        self.model.eval()
        self.rfam_low.eval()
        self.rfam_mid.eval()
        self.rfam_high.eval()
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.train()
        # for module in self.rfam_low.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.train()
        #         print("low")
        # for module in self.rfam_mid.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.train()
        #         print("mid")
        # for module in self.rfam_high.modules():
        #     if isinstance(module, torch.nn.BatchNorm2d):
        #         module.train()
        #         print("high")
        start_time = time.time()
        val_loss = AverageMeter()
        feature_norm = AverageMeter()
        outputs = []
        labels = []
        with torch.no_grad():
            previous_steps_test = (epoch-1) * len(self.test_loader)
            for batch_idx,(rgb_date, freq_data ,label, similarity_map) in tqdm(enumerate(self.test_loader),total=len(self.test_loader)):
                my_test_step = previous_steps_test + batch_idx
                rgb_date = rgb_date.to(self.device)
                freq_data = freq_data.to(self.device)
                label = label.to(self.device)
                N = label.size(0)
                # forward
                U1_low = self.model.block_1(rgb_date)
                U2_low = self.model.block_1(freq_data)
                A1_low, A2_low = self.rfam_low(U1_low, U2_low)
                x1 = U1_low * A1_low
                x2 = U2_low * A2_low

                U1_mid = self.model.block_2(x1)
                U2_mid = self.model.block_2(x2)
                A1_mid, A2_mid = self.rfam_mid(U1_mid, U2_mid)
                x1 = U1_mid * A1_mid
                x2 = U2_mid * A2_mid

                U1_high = self.model.block_3(x1)
                U2_high = self.model.block_3(x2)
                A1_high, A2_high = self.rfam_high(U1_high, U2_high)
                x1 = U1_high * A1_high
                x2 = U2_high * A2_high

                outputs_list = [(U1_low, A1_low), (U2_low, A2_low), (U1_mid, A1_mid), (U2_mid, A2_mid),
                                (U1_high, A1_high),
                                (U2_high, A2_high)]

                predicted_similarity = self.mpsm.similarity_map(outputs_list)

                predicted_similarity = predicted_similarity.to(self.device)
                predicted_label = self.model.block_4(predicted_similarity)
                label = label.to(torch.long)
                cross_loss = self.ce_loss_fn(predicted_label, label)
                self.exp.log_metric(name="test_cross_loss",value=cross_loss.item(), step=my_test_step)
                N = label.size(0)

                val_loss.update(cross_loss.item(), N)
                predicted_label = predicted_label.data.cpu().numpy()

                label = label.data.cpu().numpy()
                outputs.append(predicted_label)
                labels.append(label)

        outputs = np.concatenate(outputs)
        labels = np.concatenate(labels)

        acc, auc, tdr = evaluate(outputs,labels)

        test_metrics = {
            "test_acc": acc,
            "test_auc": auc,
            "test_tdr": tdr
        }
        self.exp.log_metrics(test_metrics, step=epoch)

        msg = '[{}][test] [epoch {}], [loss {:.5f}], [acc {:.5f}], [auc {:.5f}], [time {:.1f}], [tdr {}], [feature norm {}]'.format(
            self.exp_name, epoch, val_loss.avg, acc, auc, (time.time()-start_time)/60, tdr, feature_norm.avg)
        log_print(msg)

        # early stop
        if self.best_record['acc'] > acc and self.best_record['epoch']+5 <= epoch:
            log_print("early stop, current epoch:{}, best record:{}".format(epoch, self.best_record))
            return True
        # Save checkpoint.
        if (self.best_record['acc'] < acc and self.best_record['auc'] < auc) or (epoch+1)%10==0:
            log_print('Saving...')
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            state = {
                'model': state_dict,
                'rfam_low': self.rfam_low.state_dict(),
                'rfam_mid': self.rfam_mid.state_dict(),
                'rfam_high': self.rfam_high.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'acc': acc,
                'auc': auc,
                'epoch': epoch,
            }
            try:
                torch.save(state, '{}/epoch_{}_acc_{:.3f}_auc_{:.3f}.pth'.format(self.save_dir,epoch,acc*100,auc*100))
                # Only update best record if save was successful
                self.best_record = {'acc': acc,'auc':auc,'epoch':epoch}
            except Exception as e:
                log_print(f'Error saving checkpoint: {str(e)}')

        return False


    # @staticmethod
    # def load_and_inference(model_path, rgb_image, freq_image, device=None):
    #     """
    #     Load a saved model and perform inference on input images.
        
    #     Args:
    #         model_path (str): Path to the saved model checkpoint
    #         rgb_image (torch.Tensor): RGB input image tensor
    #         freq_image (torch.Tensor): Frequency input image tensor
    #         device (torch.device, optional): Device to run inference on. If None, uses CUDA if available
            
    #     Returns:
    #         tuple: (predicted_label, predicted_similarity)
    #             - predicted_label: Model's classification output
    #             - predicted_similarity: Similarity map prediction
    #     """
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    #     # Load checkpoint
    #     checkpoint = torch.load(model_path, map_location=device)
        
    #     # Initialize model components (you'll need to import these)
    #     model = YourModelClass()  # Replace with your actual model class
    #     rfam_low = RFAMLow()      # Replace with your actual RFAM classes
    #     rfam_mid = RFAMMid()
    #     rfam_high = RFAMHigh()
    #     mpsm = MPSM()            # Replace with your actual MPSM class
        
    #     # Load state dicts
    #     model.load_state_dict(checkpoint['model'])
    #     rfam_low.load_state_dict(checkpoint['rfam_low'])
    #     rfam_mid.load_state_dict(checkpoint['rfam_mid'])
    #     rfam_high.load_state_dict(checkpoint['rfam_high'])
        
    #     # Move models to device
    #     model = model.to(device)
    #     rfam_low = rfam_low.to(device)
    #     rfam_mid = rfam_mid.to(device)
    #     rfam_high = rfam_high.to(device)
        
    #     # Set to eval mode
    #     model.eval()
    #     rfam_low.eval()
    #     rfam_mid.eval()
    #     rfam_high.eval()
        
    #     # Prepare input data
    #     rgb_image = rgb_image.to(device)
    #     freq_image = freq_image.to(device)
        
    #     with torch.no_grad():
    #         # Forward pass through the model
    #         U1_low = model.block_1(rgb_image)
    #         U2_low = model.block_1(freq_image)
    #         A1_low, A2_low = rfam_low(U1_low, U2_low)
    #         x1 = U1_low * A1_low
    #         x2 = U2_low * A2_low

    #         U1_mid = model.block_2(x1)
    #         U2_mid = model.block_2(x2)
    #         A1_mid, A2_mid = rfam_mid(U1_mid, U2_mid)
    #         x1 = U1_mid * A1_mid
    #         x2 = U2_mid * A2_mid

    #         U1_high = model.block_3(x1)
    #         U2_high = model.block_3(x2)
    #         A1_high, A2_high = rfam_high(U1_high, U2_high)
    #         x1 = U1_high * A1_high
    #         x2 = U2_high * A2_high

    #         outputs_list = [(U1_low, A1_low), (U2_low, A2_low), 
    #                       (U1_mid, A1_mid), (U2_mid, A2_mid),
    #                       (U1_high, A1_high), (U2_high, A2_high)]

    #         predicted_similarity = mpsm.similarity_map(outputs_list)
    #         predicted_similarity = predicted_similarity.to(device)
    #         predicted_label = model.block_4(predicted_similarity)
            
    #         # Convert to numpy for easier handling
    #         predicted_label = predicted_label.cpu().numpy()
    #         predicted_similarity = predicted_similarity.cpu().numpy()
            
    #         return predicted_label, predicted_similarity

    # @staticmethod
    # def load_and_inference_batch(model_path, rgb_images, freq_images, device=None):
    #     """
    #     Load a saved model and perform inference on a batch of images.
        
    #     Args:
    #         model_path (str): Path to the saved model checkpoint
    #         rgb_images (torch.Tensor): Batch of RGB input images
    #         freq_images (torch.Tensor): Batch of frequency input images
    #         device (torch.device, optional): Device to run inference on. If None, uses CUDA if available
            
    #     Returns:
    #         tuple: (predicted_labels, predicted_similarities)
    #             - predicted_labels: Model's classification outputs for the batch
    #             - predicted_similarities: Similarity map predictions for the batch
    #     """
    #     if device is None:
    #         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
    #     # Load checkpoint
    #     checkpoint = torch.load(model_path, map_location=device)
        
    #     # Initialize model components
    #     model = YourModelClass()  # Replace with your actual model class
    #     rfam_low = RFAMLow()      # Replace with your actual RFAM classes
    #     rfam_mid = RFAMMid()
    #     rfam_high = RFAMHigh()
    #     mpsm = MPSM()            # Replace with your actual MPSM class
        
    #     # Load state dicts
    #     model.load_state_dict(checkpoint['model'])
    #     rfam_low.load_state_dict(checkpoint['rfam_low'])
    #     rfam_mid.load_state_dict(checkpoint['rfam_mid'])
    #     rfam_high.load_state_dict(checkpoint['rfam_high'])
        
    #     # Move models to device
    #     model = model.to(device)
    #     rfam_low = rfam_low.to(device)
    #     rfam_mid = rfam_mid.to(device)
    #     rfam_high = rfam_high.to(device)
    #     mpsm = mpsm.to(device)
        
    #     # Set to eval mode
    #     model.eval()
    #     rfam_low.eval()
    #     rfam_mid.eval()
    #     rfam_high.eval()
    #     mpsm.eval()
        
    #     # Prepare input data
    #     rgb_images = rgb_images.to(device)
    #     freq_images = freq_images.to(device)
        
    #     with torch.no_grad():
    #         # Forward pass through the model
    #         U1_low = model.block_1(rgb_images)
    #         U2_low = model.block_1(freq_images)
    #         A1_low, A2_low = rfam_low(U1_low, U2_low)
    #         x1 = U1_low * A1_low
    #         x2 = U2_low * A2_low

    #         U1_mid = model.block_2(x1)
    #         U2_mid = model.block_2(x2)
    #         A1_mid, A2_mid = rfam_mid(U1_mid, U2_mid)
    #         x1 = U1_mid * A1_mid
    #         x2 = U2_mid * A2_mid

    #         U1_high = model.block_3(x1)
    #         U2_high = model.block_3(x2)
    #         A1_high, A2_high = rfam_high(U1_high, U2_high)
    #         x1 = U1_high * A1_high
    #         x2 = U2_high * A2_high

    #         outputs_list = [(U1_low, A1_low), (U2_low, A2_low), 
    #                       (U1_mid, A1_mid), (U2_mid, A2_mid),
    #                       (U1_high, A1_high), (U2_high, A2_high)]

    #         predicted_similarity = mpsm.similarity_map(outputs_list)
    #         predicted_similarity = predicted_similarity.to(device)
    #         predicted_label = model.block_4(predicted_similarity)
            
    #         # Convert to numpy for easier handling
    #         predicted_label = predicted_label.cpu().numpy()
    #         predicted_similarity = predicted_similarity.cpu().numpy()
            
    #         return predicted_label, predicted_similarity