"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xcomplex<f64>>
    %4 = "stablehlo.bitcast_convert"(%2) : (tensor<2x3xcomplex<f64>>) -> tensor<2x3xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x3xcomplex<f64>>, tensor<2x3xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-1.0188829069777727,-1.7340762688128248), (-0.52944611775003347,2.8218267477233883), (-2.8165229775578471,3.3006773630951018)], [(0.77793671976434475,0.39654971695575442), (2.1567318771649968,-1.1468829935302329), (-2.6251103026761804,1.221256199025839)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%1) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[(-1.0188829069777727,-1.7340762688128248), (-0.52944611775003347,2.8218267477233883), (-2.8165229775578471,3.3006773630951018)], [(0.77793671976434475,0.39654971695575442), (2.1567318771649968,-1.1468829935302329), (-2.6251103026761804,1.221256199025839)]]> : tensor<2x3xcomplex<f64>>}> : () -> tensor<2x3xcomplex<f64>>
    "func.return"(%0) : (tensor<2x3xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

