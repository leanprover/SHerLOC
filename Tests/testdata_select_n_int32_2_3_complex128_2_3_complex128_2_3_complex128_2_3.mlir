"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5:4 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>)
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xcomplex<f64>>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<i32>}> : () -> tensor<i32>
    %8 = "stablehlo.broadcast_in_dim"(%7) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %9 = "stablehlo.compare"(%5#0, %8) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %10 = "stablehlo.constant"() <{value = dense<2> : tensor<i32>}> : () -> tensor<i32>
    %11 = "stablehlo.broadcast_in_dim"(%10) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<2x3xi32>
    %12 = "stablehlo.compare"(%5#0, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi1>
    %13 = "stablehlo.select"(%12, %5#2, %5#3) : (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3xcomplex<f64>>
    %14 = "stablehlo.select"(%9, %5#1, %13) : (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3xcomplex<f64>>
    "stablehlo.custom_call"(%14, %6) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
    "func.return"(%14) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi32>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 0, 1], [2, 0, 2]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[(-0.972899569715164,-0.19439260647629059), (1.4285494060489889,2.1200944122254861), (1.6992263098054345,5.0060033963738082)], [(-0.48134484304966618,3.0533784598258444), (3.2369772756671238,0.53969119388873188), (-2.2675026763114801,1.1807463285403068)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "stablehlo.constant"() <{value = dense<[[(0.33812031270166215,5.3939481387789412), (-1.6779673303524247,-3.6235118957269483), (2.1222088495553262,-1.2765837493591281)], [(3.2277343611739111,-0.57648162850785989), (6.0459979857659665,-1.3776613608161286), (2.324377849354657,-1.6803171534767465)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    %4 = "stablehlo.constant"() <{value = dense<[[(-4.4332885042668408,-1.3333811809215232), (-2.7715994742635184,2.7833861511184836), (2.2669882206619563,1.9938611030949978)], [(-0.36411930119219904,5.1011209546411163), (4.6669742223605475,0.6042069681966763), (-0.76515931732154541,-0.054732358304975484)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1, %2, %3, %4) : (tensor<2x3xi32>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(0.33812031270166215,5.3939481387789412), (1.4285494060489889,2.1200944122254861), (2.1222088495553262,-1.2765837493591281)], [(-0.36411930119219904,5.1011209546411163), (3.2369772756671238,0.53969119388873188), (-0.76515931732154541,-0.054732358304975484)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%0) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

