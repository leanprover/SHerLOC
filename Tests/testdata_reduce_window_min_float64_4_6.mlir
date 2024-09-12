"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xf64>
    %4 = "stablehlo.constant"() <{value = dense<0x7FF0000000000000> : tensor<f64>}> : () -> tensor<f64>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<f64>) -> tensor<f64>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<4x6xf64>, tensor<f64>) -> tensor<3x5xf64>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xf64>, tensor<3x5xf64>) -> ()
    "func.return"(%6) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-5.340512701557711, 5.6640757346528865, -0.66274802564085955, 1.4868715820375751, -1.9026789543114404, 2.057134815565874], [-3.3202575744029352, 0.3349930575415731, 2.85193019745576, 2.846085033599338, -1.7546266468898608, -3.5823896338146968], [4.1471230365198188, 2.9182955017124028, -1.5072543490921961, 5.0012649170837218, -2.6999111295778397, -0.92528331781930073], [-0.70220192940453985, 1.6593889203822185, 5.7134344530849894, -0.063442846841271355, 1.9046710707445405, 1.1704310990119449]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%1) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-5.340512701557711, -0.66274802564085955, -0.66274802564085955, -1.9026789543114404, -3.5823896338146968], [-3.3202575744029352, -1.5072543490921961, -1.5072543490921961, -2.6999111295778397, -3.5823896338146968], [-0.70220192940453985, -1.5072543490921961, -1.5072543490921961, -2.6999111295778397, -2.6999111295778397]]> : tensor<3x5xf64>}> : () -> tensor<3x5xf64>
    "func.return"(%0) : (tensor<3x5xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

