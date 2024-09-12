"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xcomplex<f64>>
    %6 = "stablehlo.select"(%4#0, %4#2, %4#1) : (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> tensor<2x3xcomplex<f64>>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
    "func.return"(%6) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %2 = "stablehlo.constant"() <{value = dense<[[(0.20488651448381795,-0.98189562308349143), (-0.34537618521175339,-3.5498123654189522), (3.4083892023097366,-0.022663425961758327)], [(3.1058804720301971,-0.915100536489034), (-3.2222121641768533,1.3265816258718024), (2.6933434704288999,0.72250984287954856)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "stablehlo.constant"() <{value = dense<[[(-0.28229432556002887,0.71646410433306551), (-0.76022405473318333,0.55355351453915591), (-0.50504892626084841,-2.0612439610208977)], [(3.0892378364363644,-2.5828326941290856), (-1.5486345575155698,-0.32902175096594749), (-0.77443543516507185,3.6713526992361958)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1, %2, %3) : (tensor<2x3xi1>, tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-0.28229432556002887,0.71646410433306551), (-0.76022405473318333,0.55355351453915591), (-0.50504892626084841,-2.0612439610208977)], [(3.0892378364363644,-2.5828326941290856), (-1.5486345575155698,-0.32902175096594749), (-0.77443543516507185,3.6713526992361958)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%0) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

