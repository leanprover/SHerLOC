/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/

/-!
# AST resulting from parsing

-/

namespace StableHLO.Parsing

abbrev FuncId := String

abbrev ValueId := String

abbrev UnusedId := String

abbrev AttrId := String

inductive Signedness where
  | signed
  | unsigned
  deriving Repr, Inhabited, Nonempty

inductive IntegerSize where
  | b2
  | b4
  | b8
  | b16
  | b32
  | b64
  deriving Repr, Inhabited, Nonempty

inductive Sign where
  | plus
  | minus
  deriving Repr, Inhabited, Nonempty

structure IntegerLiteral where
  sign : Sign
  decimal : Nat
  deriving Repr, Inhabited, Nonempty

structure FloatLiteralDecimal where
  integerPart : IntegerLiteral
  fractionalPart : IntegerLiteral
  scientificPart : IntegerLiteral
  deriving Repr, Inhabited, Nonempty

inductive FloatLiteral where
  | decimal (literal : FloatLiteralDecimal)
  | hexaDecimal (literal : Nat)
  deriving Repr, Inhabited, Nonempty

inductive BooleanLiteral where
  | true
  | false
  deriving Repr, Inhabited, Nonempty

structure ComplexLiteral where
  real : FloatLiteral
  imaginary : FloatLiteral
  deriving Repr, Inhabited, Nonempty

inductive ElementLiteral where
  | booleanLiteral (literal : BooleanLiteral)
  | floatLiteral (literal : FloatLiteral)
  | complexLiteral (literal : ComplexLiteral)
  | stringLiteral (literal : String) -- tensors can be specified with very long strings
  deriving Repr, Inhabited, Nonempty

inductive DenseLiteral where
  | denseDimension (literal : List DenseLiteral)
  | denseElements (literal : List ElementLiteral)
  deriving Repr, Inhabited, Nonempty

abbrev TensorLiteral := DenseLiteral

inductive ComparisonDirection where
  | eq
  | ne
  | ge
  | gt
  | le
  | lt
  deriving Repr, Inhabited, Nonempty

inductive CompareType where
  | float
  | totalOrder
  | signed
  | unsigned
  deriving Repr, Inhabited, Nonempty

inductive PrecisionConfig where
  | default
  | high
  | highest
  deriving Repr, Inhabited, Nonempty

inductive FftType where
  | fft
  | ifft
  | rfft
  | irfft
  deriving Repr, Inhabited, Nonempty

inductive ChannelType where
  | deviceToDevice
  | hostToDevice
  deriving Repr, Inhabited, Nonempty

inductive RngDistribution where
  | uniform
  | normal
  deriving Repr, Inhabited, Nonempty

inductive RngAlgorithm where
  | default
  | threeFry
  | philox
  deriving Repr, Inhabited, Nonempty

inductive TransposeA where
  | noTranspose
  | transpose
  | adjoint
  deriving Repr, Inhabited, Nonempty

inductive EnumLiteral where
  | comparisonDirection (enum : ComparisonDirection)
  | compareType (enum : CompareType)
  | precisionConfig (enum : PrecisionConfig)
  | fftType (enum : FftType)
  | channelType (enum : ChannelType)
  | rngDistribution (enum : RngDistribution)
  | rngAlgorithm (enum : RngAlgorithm)
  | transposeA (enum : TransposeA)
  deriving Repr, Inhabited, Nonempty

inductive ArrayLiteral where
  | array64 (literal : List IntegerLiteral)
  | array1 (literal : List BooleanLiteral)
  deriving Repr, Inhabited, Nonempty

inductive StableHLORecordFieldValue where
  | one (literal : Nat)
  | many (literal : List Nat)
  deriving Repr, Inhabited, Nonempty

structure StableHLORecordField where
  name : String
  value : StableHLORecordFieldValue
  deriving Repr, Inhabited, Nonempty

inductive ConvolutionMode where
  | i
  | o
  | f
  | one
  | b
  | zero
  | two
  deriving Repr, Inhabited, Nonempty

structure Convolution where
  lhs : List ConvolutionMode
  rhs : List ConvolutionMode
  result : List ConvolutionMode
  deriving Repr, Inhabited, Nonempty

inductive Literal where
  | enum (literal : EnumLiteral)
  | element (literal : ElementLiteral)
  | tensor (literal : TensorLiteral)
  | string (literal : String)
  | stableHLORecord (literal : List StableHLORecordField)
  | convolution (literal : Convolution)

  | special
  | use_global_device_ids
  | array (literal : ArrayLiteral)
  | experiment1 (literal : List FuncId)
  | experiment2 (literal : List (List FuncId))
  | experiment3 (literal : List EnumLiteral)
  | func (literal : FuncId)
  deriving Repr, Inhabited, Nonempty

structure IntegerType where
  sign : Signedness
  size : IntegerSize
  deriving Repr, Inhabited, Nonempty

inductive FloatType where
  | f8E4M3FN
  | f8E5M2
  | f8E4M3FNUZ
  | f8E5M2FNUZ
  | f8E4M3B11FNUZ
  | bf16
  | f16
  | f32
  | f64
  | tf32
  deriving Repr, Inhabited, Nonempty

inductive NumberType where
  | integerType (type : IntegerType)
  | floatType (type: FloatType)
  deriving Repr, Inhabited, Nonempty

inductive ComplexType where
  | f32
  | f64
  deriving Repr, Inhabited, Nonempty

inductive TensorElementType where
  | booleanType
  | integerType (t : IntegerType)
  | floatType (t: FloatType)
  | complexType (t: ComplexType)
  deriving Repr, Inhabited, Nonempty

structure QuantizationParameter where
  quantizationScale : FloatLiteral
  quantizationZeroPoint: IntegerLiteral
  deriving Repr, Inhabited, Nonempty

structure QuantizationBasics where
  quantizationStorageType : IntegerType
  quantizationStorageMinMax : Option (IntegerLiteral × IntegerLiteral)
  quantizationExpressedType : FloatType
  quantizationDimension : Option IntegerLiteral
  deriving Repr, Inhabited, Nonempty

structure QuantizedTensorElementType where
  quantizationBasics : QuantizationBasics
  quantizationParameters : List QuantizationParameter
  deriving Repr, Inhabited, Nonempty

inductive TensorElementTypeGen where
  | classic (t : TensorElementType)
  | quantized (t : QuantizedTensorElementType)
  deriving Repr, Inhabited, Nonempty

inductive DimensionSize where
  | known (size : Nat)
  | unknown
  deriving Repr, Inhabited, Nonempty

structure TensorType where
  shape : List DimensionSize
  tensorElementTypeGen : TensorElementTypeGen
  deriving Repr, Inhabited, Nonempty

inductive ValueType where
  | tensorType (tensor : TensorType)
  | tokenType
  | tupleType (elements : List ValueType)
  deriving Repr, Inhabited, Nonempty

structure FunctionType where
  domain : List ValueType
  range : List ValueType
  deriving Repr, Inhabited, Nonempty

inductive NonValueType where
  | tensorElementType (t : TensorElementType)
  | quantizedTensorElementType (t: QuantizedTensorElementType)
  | functionType (t : FunctionType)
  | stringType
  deriving Repr, Inhabited, Nonempty

inductive SType where
  | valueType (t : ValueType)
  | nonValueType (t : NonValueType)
  deriving Repr, Inhabited, Nonempty

structure Constant where
  literal : Literal
  typ : Option SType
  deriving Repr, Inhabited, Nonempty

structure Attribute where
  id : AttrId
  constant : Constant
  deriving Repr, Inhabited, Nonempty

structure FuncInput where
  id : FuncId
  typ : ValueType
  deriving Repr, Inhabited, Nonempty

mutual

  inductive InputFunc where
    | mk
      (funcInputs : List FuncInput)
      (body : List Operation)
    deriving Repr, Inhabited, Nonempty

  inductive Operation where
    | stablehlo
      (name : String)
      (inputValues : List ValueId)
      (inputFunctions : List InputFunc)
      (inputAttributes : List Attribute)
      (outputs : List ValueId)
      (signature : FunctionType)
    | return
      (operands : List ValueId)
      (signature : FunctionType)
    | call
      (callee : FuncId)
      (inputValues : List ValueId)
      (outputs : List ValueId)
      (signature : FunctionType)
    deriving Repr, Inhabited, Nonempty

end

structure Function where
  funcId : FuncId
  funcArgAttrs : List (List Attribute)
  funcResAttrs : List (List Attribute)
  funcType : FunctionType
  funcBody : InputFunc
  deriving Repr, Inhabited, Nonempty

structure Module where
  modId : Option FuncId
  modAttrs : List Attribute
  modFuncs : List Function
  deriving Repr, Inhabited, Nonempty

def Program := List Module
  deriving Repr, Inhabited, Nonempty

end StableHLO.Parsing
