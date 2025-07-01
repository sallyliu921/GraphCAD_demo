# -*- coding: UTF-8 -*-
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from init import *
from LoadData import LoadData, restore_curve
from model import *
from utils import hetero_collate_fn, evaluate_metrics
from tqdm import tqdm
import pandas as pd

def test(start_time, test_loader):
    # Load model
    model = torch.load(model_path, weights_only=False)

    model.eval()  
    test_loss = 0.0
    num_batches = 0
    
    all_predicted_delays, all_target_delays = [], []
    all_error_rate_list = []
    all_key_entries, all_driver_pins, all_receiver_pins = [], [], []    
   
    with torch.no_grad(): 
        for batch in tqdm(test_loader, desc=f'Testing'):
            batch_key_entry = [ke.decode('utf-8') for ke in batch['key_entry']]
            batch_driver_pin = [dp.decode('utf-8') for dp in batch['driver_pin']]
            batch_receiver_pin = [rp.decode('utf-8') for rp in batch['receiver_pin']]
            batch_x_dict = [{k: v.to(device) for k, v in x_dict.items()} for x_dict in batch['x_dict']]
            batch_edge_index_dict = [{k: v.to(device) for k, v in edge_index_dict.items()} for edge_index_dict in batch['edge_index_dict']]
            batch_p_data = [[path_data.to(device) for path_data in p_data_list] for p_data_list in batch['p_data']]
            target_delay = batch['delay'].to(device)
            

            delay_prediction_net, delay_prediction_trans = model(batch_x_dict, batch_edge_index_dict, batch_p_data)
            delay_prediction = torch.stack((delay_prediction_net.squeeze(), delay_prediction_trans.squeeze()), dim=-1)
            
            loss = combined_loss(delay_prediction, target_delay)
                
            test_loss += loss.item()
            num_batches += 1
            
            # Deal with the predicted delay value
            all_key_entries.extend(batch_key_entry)
            all_driver_pins.extend(batch_driver_pin)
            all_receiver_pins.extend(batch_receiver_pin)
            all_target_delays.extend(target_delay.cpu().numpy().tolist())
            
            delay_prediction_list = delay_prediction.cpu().numpy().tolist()

            if not isinstance(delay_prediction_list[0], list):
                all_predicted_delays.extend([delay_prediction_list])
            else:
                all_predicted_delays.extend(delay_prediction_list)
        
            if delay_prediction.dim() == 1:
                delay_prediction_trans = torch.unsqueeze(delay_prediction[1], 0) 
            else:
                delay_prediction_trans = delay_prediction[:,1]
                
        test_loss /= num_batches
        print("*******Test Loss: {:.4f}****".format(test_loss))

        data_pd = {
            'key_entry': all_key_entries,
            'input_pin': all_driver_pins,
            'output_pin': all_receiver_pins,
            'GNNSI Predicted Net Delay': [d[0] for d in all_predicted_delays],
            'GNNSI Predicted Trans': [d[1] for d in all_predicted_delays],
        }
 
        df = pd.DataFrame(data_pd)
        df.to_excel(os.path.join(output_path, 'GNNSI_result.xlsx'), index=False)
        print("Data saved to: ", os.path.join(output_path, 'GNNSI_result.xlsx'))
        
        metrics = evaluate_metrics(all_predicted_delays, all_target_delays)
        for name, value in metrics.items():
            print(f"{name}: {value}")

        end_time = datetime.datetime.now()  
        print("End Time:", end_time.strftime("%Y-%m-%d %H:%M"))

        total_time = end_time - start_time  
        hours = total_time.seconds // 3600  
        minutes = (total_time.seconds % 3600) // 60  
        print("Total Time:", hours, "hours", minutes, "minutes")

def main():
    # Data Loader
    start_time = datetime.datetime.now()
    print("Start Time:", start_time.strftime("%m-%d %H:%M"))
    all_data = LoadData(benchmark_path, dataset_path)
    graph_num = len(all_data)
    print("Dataset Num:", graph_num)

    test_dataset = GraphDataset(all_data, use_composable=False, shuffle=False)    
    num_workers = 4
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=hetero_collate_fn)
    
    test(start_time, test_loader)
    
if __name__ == "__main__":
    main()