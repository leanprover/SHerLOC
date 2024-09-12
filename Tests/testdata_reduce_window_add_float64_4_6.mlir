"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf64>
    %4 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<f64>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<4x6xf64>, tensor<f64>) -> tensor<3x5xf64>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    "func.return"(%6) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.5324336566911518, 2.1188302422914136, -4.3439614113572986, -0.38660422134149153, 4.8135446144226997, 3.3302138332773001], [-0.40416879691901597, -3.4819809639599359, -0.45688435409191519, 4.0382113839432625, -3.6605793473452435, 6.3495109586805896], [-4.2065526539024756, 1.2729479381498401, -4.0316152340148594, 3.1785554961636042, -1.7193908982915442, -2.9703609618433493], [-1.5103160091056655, 2.4593906236248095, 5.2904055457663386, -4.8652568030895749, -4.7078295773267325, -3.7866229428737368]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%1) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3.2997531752786902, -6.1639964871177364, -1.1492386028474426, 4.8045724296792267, 10.832690059035347], [-6.8197544766315881, -6.6975326139168709, 2.7282672920000919, 1.836796634470079, -2.0008202487995472], [-1.9845301012334922, 4.9911288735261286, -0.42791099517449194, -8.1139217825442476, -13.184204380335363]]> : tensor<3x5xf64>}> : () -> tensor<3x5xf64>
    "func.return"(%0) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

