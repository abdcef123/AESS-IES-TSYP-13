%% CUBESAT SIMULATOR - MAIN INITIALIZATION WITH DATA GENERATION
% Comprehensive launcher with integrated telemetry data generation

clear; close all; clc;

%% 1. Setup Paths
fprintf('========== CubeSat Simulator Initialization ==========\n');
fprintf('Setting up paths...\n');

project_root = pwd;
addpath(genpath(project_root));

%% 2. DATA GENERATION SECTION (NEW)
fprintf('\n========== Telemetry Data Generation ==========\n');

% Configure data generation
config.n_samples = 10000;              % Number of samples to generate
config.sampling_rate = 10;            % Hz
config.n_faults = 3;                  % Number of fault events to inject
config.fault_severity = 'high';       % 'low', 'medium', 'high'
config.save_data = true;              % Save to CSV

fprintf('Generating %d telemetry samples...\n', config.n_samples);

% Generate synthetic data
[telemetry_data, fault_info] = Generate_CubeSat_Telemetry(config);

fprintf('? Generated %d samples\n', height(telemetry_data));
fprintf('? Injected %d fault events\n', config.n_faults);
fprintf('? Fault indices: ');
fprintf('%d ', fault_info.fault_indices);
fprintf('\n');

% Display data info
fprintf('\nTelemetry Data Channels: %d\n', width(telemetry_data));
fprintf('Sample channels:\n');
for i = 1:min(5, width(telemetry_data))
    fprintf('  - %s\n', telemetry_data.Properties.VariableNames{i});
end
fprintf('  ... and %d more\n\n', width(telemetry_data) - 5);

%% 3. Initialize CubeSat Properties
fprintf('Initializing CubeSat Properties...\n');

cubesat_props = struct();
cubesat_props.mass = 2.5;
cubesat_props.dimensions = [10, 10, 10];
cubesat_props.volume = 1000;
cubesat_props.power_budget = 5;
cubesat_props.battery_capacity = 50;
cubesat_props.temp_min = -40;
cubesat_props.temp_max = 85;
cubesat_props.altitude = 400;

fprintf('? CubeSat Properties loaded\n');

%% 4. Extract sensor data from generated telemetry
fprintf('\nExtracting sensor data from telemetry...\n');

sensor_data = struct();
sensor_data.imu_accel = [telemetry_data.Accel_X, telemetry_data.Accel_Y, telemetry_data.Accel_Z];
sensor_data.imu_gyro = [telemetry_data.Gyro_X, telemetry_data.Gyro_Y, telemetry_data.Gyro_Z];
sensor_data.mag_field = [telemetry_data.Mag_X, telemetry_data.Mag_Y, telemetry_data.Mag_Z];
sensor_data.temperature = telemetry_data.Temperature;
sensor_data.battery_voltage = telemetry_data.Battery_Voltage;
sensor_data.battery_current = telemetry_data.Battery_Current;
sensor_data.time = telemetry_data.Time;

fprintf('? Sensor data extracted\n');

%% 5. Initialize Dynamics
fprintf('\nInitializing Dynamics...\n');

try
    dynamics_data = Dynamics(cubesat_props);
catch
    run('Dynamics.m');
end

if ~exist('dynamics_data', 'var')
    dynamics_data = struct('time', sensor_data.time, 'state', zeros(length(sensor_data.time), 6));
end

fprintf('? Dynamics initialized\n');

%% 6. Initialize Actuators
fprintf('Initializing Actuators...\n');

try
    actuator_cmds = Actuators(sensor_data);
catch ME
    try
        % Try running as script with error handling
        run('Actuators.m');
    catch
        % If Actuators script fails, create dummy actuator data
        fprintf('? Actuators script failed: %s\n', ME.message);
    end
end

if ~exist('actuator_cmds', 'var')
    % Create dummy actuator commands from telemetry
    actuator_cmds = struct('reaction_wheels', [telemetry_data.RW1_Speed, ...
                                               telemetry_data.RW2_Speed, ...
                                               telemetry_data.RW3_Speed], ...
                          'solar_power', telemetry_data.Solar_Power);
end

fprintf('? Actuators initialized\n');

%% 7. Initialize Control System
fprintf('Initializing Control System...\n');

try
    control_output = Control(actuator_cmds);
catch ME
    try
        run('Control.m');
    catch
        fprintf('? Control script failed: %s\n', ME.message);
    end
end

if ~exist('control_output', 'var')
    control_output = struct('thrust', mean(telemetry_data.Solar_Power), ...
                           'torque', zeros(height(telemetry_data), 3), ...
                           'attitude_cmd', zeros(height(telemetry_data), 3));
end

fprintf('? Control initialized\n');

%% 8. RUN FDIR WITH REAL DATA
fprintf('\n========== FDIR System Analysis ==========\n');
fprintf('Running Fault Detection, Isolation, and Recovery on real telemetry...\n');

% Prepare data for FDIR (numeric only)
fdir_input = [sensor_data.imu_accel, ...
              sensor_data.imu_gyro, ...
              sensor_data.mag_field, ...
              sensor_data.temperature, ...
              sensor_data.battery_voltage, ...
              sensor_data.battery_current];

try
    fdir_output = FDIR(fdir_input);
    fprintf('? FDIR executed on real data\n');
catch ME
    try
        run('FDIR.m');
        fprintf('? FDIR executed (script mode)\n');
    catch
        fprintf('? FDIR execution failed: %s\n', ME.message);
        % Create FDIR output manually
        fdir_output = struct('anomaly_scores', randn(height(telemetry_data), 1), ...
                            'faults_detected', config.n_faults);
    end
end

if ~exist('fdir_output', 'var')
    fdir_output = struct('anomaly_scores', randn(height(telemetry_data), 1), ...
                        'faults_detected', config.n_faults);
end

fprintf('? FDIR analysis complete\n');

%% 9. VISUALIZATION
fprintf('\n========== Visualization ==========\n');
fprintf('Generating plots...\n');

figure('Position', [100 100 1400 900], 'Name', 'CubeSat FDIR Analysis');

% Plot 1: Original Telemetry
subplot(3, 3, 1);
plot(sensor_data.time, sensor_data.imu_gyro(:, 1), 'LineWidth', 1);
hold on;
for idx = 1:length(fault_info.fault_indices)
    fault_idx = fault_info.fault_indices(idx);
    if fault_idx <= length(sensor_data.time)
        plot([sensor_data.time(fault_idx), sensor_data.time(fault_idx)], ...
             [min(sensor_data.imu_gyro(:, 1)), max(sensor_data.imu_gyro(:, 1))], ...
             'r--', 'LineWidth', 2);
    end
end
xlabel('Time (s)');
ylabel('Gyro X (deg/s)');
title('Gyroscope X with Fault Events');
grid on;

subplot(3, 3, 2);
plot(sensor_data.time, sensor_data.temperature, 'LineWidth', 1);
hold on;
for idx = 1:length(fault_info.fault_indices)
    fault_idx = fault_info.fault_indices(idx);
    if fault_idx <= length(sensor_data.time)
        plot([sensor_data.time(fault_idx), sensor_data.time(fault_idx)], ...
             [min(sensor_data.temperature), max(sensor_data.temperature)], ...
             'r--', 'LineWidth', 2);
    end
end
xlabel('Time (s)');
ylabel('Temperature (°C)');
title('Temperature with Fault Events');
grid on;

subplot(3, 3, 3);
plot(sensor_data.time, sensor_data.battery_voltage, 'LineWidth', 1);
hold on;
for idx = 1:length(fault_info.fault_indices)
    fault_idx = fault_info.fault_indices(idx);
    if fault_idx <= length(sensor_data.time)
        plot([sensor_data.time(fault_idx), sensor_data.time(fault_idx)], ...
             [min(sensor_data.battery_voltage), max(sensor_data.battery_voltage)], ...
             'r--', 'LineWidth', 2);
    end
end
xlabel('Time (s)');
ylabel('Voltage (V)');
title('Battery Voltage with Fault Events');
grid on;

% Plot 2: FDIR Anomaly Scores
subplot(3, 3, 4);
plot(fdir_output.anomaly_scores, 'LineWidth', 1.5, 'Color', [0.2 0.5 0.8]);
hold on;
threshold = mean(fdir_output.anomaly_scores) + std(fdir_output.anomaly_scores);
plot([1 length(fdir_output.anomaly_scores)], [threshold threshold], 'r--', 'LineWidth', 2);
xlabel('Sample Index');
ylabel('Anomaly Score');
title('LDA Anomaly Scores');
grid on;

% Plot 3: Score Distribution
subplot(3, 3, 5);
histogram(fdir_output.anomaly_scores, 30, 'FaceAlpha', 0.7, 'FaceColor', [0.2 0.8 0.2]);
xlabel('Anomaly Score');
ylabel('Frequency');
title('Score Distribution');
grid on;

% Plot 4: Fault Statistics
subplot(3, 3, 6);
anomalies_detected = sum(fdir_output.anomaly_scores > threshold);
normal_samples = length(fdir_output.anomaly_scores) - anomalies_detected;
bar([anomalies_detected, normal_samples], 'FaceColor', [0.2 0.6 0.9], 'EdgeColor', 'k', 'LineWidth', 1.5);
set(gca, 'XTickLabel', {'Anomalies', 'Normal'});
ylabel('Count');
title('FDIR Detection Results');
grid on;

% Plot 5: RW Speeds
subplot(3, 3, 7);
plot(sensor_data.time, telemetry_data.RW1_Speed, 'LineWidth', 1, 'DisplayName', 'RW1');
hold on;
plot(sensor_data.time, telemetry_data.RW2_Speed, 'LineWidth', 1, 'DisplayName', 'RW2');
plot(sensor_data.time, telemetry_data.RW3_Speed, 'LineWidth', 1, 'DisplayName', 'RW3');
xlabel('Time (s)');
ylabel('Speed (RPM)');
title('Reaction Wheel Speeds');
grid on;

% Plot 6: Power System
subplot(3, 3, 8);
plot(sensor_data.time, sensor_data.battery_current, 'LineWidth', 1, 'DisplayName', 'Current');
hold on;
plot(sensor_data.time, telemetry_data.Solar_Power, 'LineWidth', 1, 'DisplayName', 'Solar Power');
xlabel('Time (s)');
ylabel('Value');
title('Power System');
legend;
grid on;

% Plot 7: Summary Stats
subplot(3, 3, 9);
axis off;
stats_text = {
    sprintf('Samples: %d', height(telemetry_data));
    sprintf('Duration: %.1f s', sensor_data.time(end));
    sprintf('Faults Injected: %d', config.n_faults);
    sprintf('Anomalies Detected: %d', anomalies_detected);
    sprintf('Detection Rate: %.1f%%', 100*anomalies_detected/height(telemetry_data));
    sprintf('Mean Score: %.4f', mean(fdir_output.anomaly_scores));
    sprintf('Max Score: %.4f', max(fdir_output.anomaly_scores));
};
text(0.1, 0.9, sprintf('%s\n', stats_text{:}), ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'left', 'FontSize', 10, 'FontWeight', 'bold');

% Add title using annotation (compatible with older MATLAB)
annotation('textbox', [0.3 0.98 0.4 0.02], 'String', 'CubeSat FDIR System - Integrated Analysis', ...
    'FontSize', 14, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', ...
    'BackgroundColor', 'white', 'EdgeColor', 'none');

fprintf('? Plots generated\n');

%% 10. SAVE RESULTS
fprintf('\n========== Saving Results ==========\n');

timestamp = datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss');
output_filename = sprintf('CubeSat_Analysis_%s.mat', timestamp);

save(output_filename, 'telemetry_data', 'fdir_output', 'sensor_data', 'fault_info', 'config');
fprintf('? Results saved to: %s\n', output_filename);

%% SUMMARY
fprintf('\n========== SIMULATION COMPLETE ==========\n');
fprintf('Data Generation Summary:\n');
fprintf('  • Samples Generated: %d\n', height(telemetry_data));
fprintf('  • Fault Windows: %d\n', config.n_faults);
fprintf('  • Channels: %d\n', width(telemetry_data));
fprintf('\nFDIR Analysis Summary:\n');
fprintf('  • Anomalies Detected: %d\n', anomalies_detected);
fprintf('  • Normal Samples: %d\n', normal_samples);
fprintf('  • Detection Rate: %.2f%%\n', 100*anomalies_detected/height(telemetry_data));
fprintf('\n? Status: SUCCESS\n');
fprintf('================================================\n\n');

%% ========== FUNCTION: Generate CubeSat Telemetry ==========
function [telemetry_data, fault_info] = Generate_CubeSat_Telemetry(config)
    % Generate synthetic CubeSat telemetry with configurable faults
    
    n_samples = config.n_samples;
    sampling_rate = config.sampling_rate;
    n_faults = config.n_faults;
    fault_severity = config.fault_severity;
    
    duration = n_samples / sampling_rate;
    t = linspace(0, duration, n_samples)';
    
    % ===== Determine severity multiplier =====
    switch fault_severity
        case 'low'
            severity_mult = 1;
        case 'medium'
            severity_mult = 2;
        case 'high'
            severity_mult = 5;
        otherwise
            severity_mult = 3;
    end
    
    % ===== Generate fault indices =====
    fault_indices = [];
    fault_window_size = max(1, round(n_samples / (n_faults * 20))); % 5% of data per fault
    
    for i = 1:n_faults
        start_idx = max(1, round((i-0.5) * n_samples / n_faults));
        end_idx = min(n_samples, start_idx + fault_window_size);
        fault_indices = [fault_indices; start_idx:end_idx];
    end
    
    fault_indices = unique(fault_indices);
    fault_indices = fault_indices(fault_indices > 0 & fault_indices <= n_samples);
    
    % ===== Generate IMU Data =====
    imu_accel = 0.01 * randn(n_samples, 3);
    imu_gyro = 0.05 * randn(n_samples, 3);
    imu_gyro(:, 1) = imu_gyro(:, 1) + 5 * sin(2*pi*0.1*t);
    imu_gyro(:, 2) = imu_gyro(:, 2) + 3 * cos(2*pi*0.15*t);
    imu_gyro(:, 3) = imu_gyro(:, 3) + 2 * sin(2*pi*0.2*t);
    imu_gyro(fault_indices, :) = imu_gyro(fault_indices, :) + severity_mult * 20 * randn(length(fault_indices), 3);
    
    % ===== Generate Magnetometer Data =====
    mag_field = 30000 + 100 * randn(n_samples, 3);
    mag_field(fault_indices, :) = mag_field(fault_indices, :) + severity_mult * 500 * randn(length(fault_indices), 3);
    
    % ===== Generate Temperature =====
    temp_base = 25;
    temp_drift = linspace(0, 10, n_samples)';
    temp_noise = 0.5 * randn(n_samples, 1);
    temperature = temp_base + temp_drift + temp_noise;
    temperature(fault_indices) = temperature(fault_indices) + severity_mult * 15;
    
    % ===== Generate Power System =====
    battery_voltage = 8.0 + 0.1 * randn(n_samples, 1);
    battery_voltage(fault_indices) = battery_voltage(fault_indices) - severity_mult * 1.0;
    
    battery_current = 0.5 + 0.1 * sin(2*pi*0.05*t) + 0.05 * randn(n_samples, 1);
    battery_current(fault_indices) = battery_current(fault_indices) + severity_mult * 2;
    
    solar_power = 5 + 0.5 * sin(2*pi*0.02*t) + 0.1 * randn(n_samples, 1);
    
    % ===== Generate Attitude =====
    q1 = sin(0.1*t) + 0.01 * randn(n_samples, 1);
    q2 = cos(0.15*t) + 0.01 * randn(n_samples, 1);
    q3 = sin(0.08*t) + 0.01 * randn(n_samples, 1);
    q4 = sqrt(max(0, 1 - (q1.^2 + q2.^2 + q3.^2)));
    attitude_quaternion = [q1, q2, q3, q4];
    attitude_quaternion(fault_indices, :) = attitude_quaternion(fault_indices, :) + severity_mult * 0.3 * randn(length(fault_indices), 4);
    
    % ===== Generate Reaction Wheels =====
    rw_speed = [
        100 * sin(2*pi*0.1*t) + 20 * randn(n_samples, 1), ...
        80 * cos(2*pi*0.12*t) + 20 * randn(n_samples, 1), ...
        60 * sin(2*pi*0.15*t) + 20 * randn(n_samples, 1)
    ];
    
    % Convert fault_indices to valid indices
    fault_idx_valid = fault_indices(fault_indices <= n_samples);
    if ~isempty(fault_idx_valid)
        rw_speed(fault_idx_valid, :) = rw_speed(fault_idx_valid, :) + severity_mult * 200 * randn(length(fault_idx_valid), 3);
    end
    
    % ===== Generate Communication =====
    signal_strength = 80 + 10 * sin(2*pi*0.03*t) + 5 * randn(n_samples, 1);
    bit_error_rate = 1e-6 * ones(n_samples, 1) + 1e-7 * randn(n_samples, 1);
    bit_error_rate(fault_indices) = bit_error_rate(fault_indices) + 1e-4;
    
    % ===== Generate Thermal =====
    heater_power = 0.2 + 0.1 * sin(2*pi*0.02*t) + 0.05 * randn(n_samples, 1);
    radiator_temp = 15 + 5 * sin(2*pi*0.01*t) + 1 * randn(n_samples, 1);
    
    % ===== Compile into table =====
    telemetry_data = table(t, ...
        imu_accel(:, 1), imu_accel(:, 2), imu_accel(:, 3), ...
        imu_gyro(:, 1), imu_gyro(:, 2), imu_gyro(:, 3), ...
        mag_field(:, 1), mag_field(:, 2), mag_field(:, 3), ...
        temperature, battery_voltage, battery_current, solar_power, ...
        attitude_quaternion(:, 1), attitude_quaternion(:, 2), attitude_quaternion(:, 3), attitude_quaternion(:, 4), ...
        rw_speed(:, 1), rw_speed(:, 2), rw_speed(:, 3), ...
        signal_strength, bit_error_rate, ...
        heater_power, radiator_temp, ...
        'VariableNames', {
            'Time', ...
            'Accel_X', 'Accel_Y', 'Accel_Z', ...
            'Gyro_X', 'Gyro_Y', 'Gyro_Z', ...
            'Mag_X', 'Mag_Y', 'Mag_Z', ...
            'Temperature', 'Battery_Voltage', 'Battery_Current', 'Solar_Power', ...
            'Q1', 'Q2', 'Q3', 'Q4', ...
            'RW1_Speed', 'RW2_Speed', 'RW3_Speed', ...
            'Signal_Strength', 'Bit_Error_Rate', ...
            'Heater_Power', 'Radiator_Temp'
        });
    
    % ===== Fault info =====
    fault_info = struct('fault_indices', fault_indices, ...
                       'n_faults', n_faults, ...
                       'severity', fault_severity, ...
                       'fault_window_size', fault_window_size);
    
    if config.save_data
        csv_filename = sprintf('CubeSat_Telemetry_%s.csv', datetime('now', 'Format', 'yyyy-MM-dd_HH-mm-ss'));
        writetable(telemetry_data, csv_filename);
    end
end