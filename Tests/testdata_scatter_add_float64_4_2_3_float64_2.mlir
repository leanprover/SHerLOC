"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[3, 2]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3xf64>, tensor<2xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [0, 2], scatter_dims_to_operand_dims = [0, 2]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<4x2x3xf64>, tensor<2xi64>, tensor<2xf64>) -> tensor<4x2x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3xf64>, tensor<4x2x3xf64>) -> ()
    "func.return"(%6) : (tensor<4x2x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3xf64>, tensor<2xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[4.0628851850084056, -2.1789947508853063, -0.79475880803133669], [-1.1962013509265845, -3.0588897338088721, -1.2548185136818808]], [[-0.045384383318880547, 2.8473003493227029, 2.3365440518420506], [-3.3190170056728987, 0.16356163108892877, 3.2911112172773853]], [[-1.4502748355409238, -3.6289788327582162, 1.2305928295093609], [2.6213756045044549, 2.0697478494643038, 4.8013330951367248]], [[-1.2788618627291946, 4.4414958748056899, 0.78695170664571068], [0.5016198221803233, 7.6313637227526065, 0.67811813096908546]]]> : tensor<4x2x3xf64>}> : () -> tensor<4x2x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[1.0604575531757425, -1.9256271883077405]> : tensor<2xf64>}> : () -> tensor<2xf64>
    "func.return"(%1, %2) : (tensor<4x2x3xf64>, tensor<2xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[4.0628851850084056, -2.1789947508853063, -0.79475880803133669], [-1.1962013509265845, -3.0588897338088721, -1.2548185136818808]], [[-0.045384383318880547, 2.8473003493227029, 2.3365440518420506], [-3.3190170056728987, 0.16356163108892877, 3.2911112172773853]], [[-1.4502748355409238, -3.6289788327582162, 1.2305928295093609], [2.6213756045044549, 2.0697478494643038, 4.8013330951367248]], [[-1.2788618627291946, 4.4414958748056899, 1.8474092598214531], [0.5016198221803233, 7.6313637227526065, -1.2475090573386551]]]> : tensor<4x2x3xf64>}> : () -> tensor<4x2x3xf64>
    "func.return"(%0) : (tensor<4x2x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

