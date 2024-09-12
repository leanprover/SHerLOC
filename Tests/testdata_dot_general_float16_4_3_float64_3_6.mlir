"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x3xf16>, tensor<3x6xf64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x6xf64>
    %5 = "stablehlo.convert"(%3#0) : (tensor<4x3xf16>) -> tensor<4x3xf64>
    %6 = "stablehlo.convert"(%3#1) : (tensor<3x6xf64>) -> tensor<3x6xf64>
    %7 = "stablehlo.dot_general"(%5, %6) <{dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]>}> : (tensor<4x3xf64>, tensor<3x6xf64>) -> tensor<4x6xf64>
    "stablehlo.custom_call"(%7, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x6xf64>, tensor<4x6xf64>) -> ()
    "func.return"(%7) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x3xf16>, tensor<3x6xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3.595700e+00, 3.619140e+00, -5.261720e+00], [1.973880e-01, -1.737300e+00, -2.189940e-01], [-2.433590e+00, 7.119140e-01, -4.019530e+00], [9.355460e-01, 2.970700e+00, -1.253910e+00]]> : tensor<4x3xf16>}> : () -> tensor<4x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[1.7644197472699148, 1.7043742348703241, 1.5261589163458105, 0.29420515636325822, 2.3672846682690594, -2.6927424430293074], [-0.40181932535446785, -4.7279690310955527, 4.3225875343381333, 1.5654165620697107, -5.338042688148974, -3.05218271806035], [-0.6860180832954107, -3.7113500994186488, 4.2941046651038839, 0.36092700378597814, 2.5590305312603814, -2.7780545373475642]]> : tensor<3x6xf64>}> : () -> tensor<3x6xf64>
    "func.return"(%1, %2) : (tensor<4x3xf16>, tensor<3x6xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-4.1889360316560662, -3.7115281507470428, -12.4379332649491, 2.7084918915058407, -41.296078958279011, 13.253365595049736], [1.1965912855483649, 9.3631091878288916, -8.1487903552448131, -2.7405739524327122, 9.1806667569880531, 5.3794347843088435], [-1.8224705737657463, 7.4042255783609887, -17.897027840399264, -1.0522911362221743, -19.847340019194167, 15.546626408482044], [1.3172138174878851, -7.7971853007161682, 8.8845128235571309, 4.4730619616707674, -16.851818698521239, -8.1028955691478401]]> : tensor<4x6xf64>}> : () -> tensor<4x6xf64>
    "func.return"(%0) : (tensor<4x6xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

